#!/usr/bin/env python3
"""
Inference + evaluation for PAFC Top-10 QLoRA adapters trained with LLaMA-Factory.

Works with:
  - examples/train_lora/pafc_qwen25_qlora_top10.yaml
  - examples/train_lora/pafc_llama_instruct3.1_qlora_top10.yaml

Important:
  The training YAML uses 4-bit QLoRA, but for merging we load the base model
  in bf16/fp16, then load the trained adapter and call merge_and_unload().
  Do not load the base model in 4-bit when you want to merge and save/infer
  from the merged full model.

Example:
  python infer_pafc_top10_qlora_merge_eval.py \
    --config examples/train_lora/pafc_qwen25_qlora_top10.yaml \
    --test_file data/pafc_top10_eval.json \
    --out_dir eval_outputs/pafc_qwen25_qlora_top10
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = [
    "SUPPORTED",
    "REFUTED",
    "NOT ENOUGH EVIDENCE",
    "CONFLICTING",
]

LABEL_ALIASES = {
    "SUPPORTED": "SUPPORTED",
    "SUPPORT": "SUPPORTED",
    "SUPPORTS": "SUPPORTED",
    "TRUE": "SUPPORTED",
    "REFUTED": "REFUTED",
    "REFUTE": "REFUTED",
    "REFUTES": "REFUTED",
    "FALSE": "REFUTED",
    "NOT ENOUGH EVIDENCE": "NOT ENOUGH EVIDENCE",
    "NO ENOUGH EVIDENCE": "NOT ENOUGH EVIDENCE",
    "INSUFFICIENT EVIDENCE": "NOT ENOUGH EVIDENCE",
    "NEI": "NOT ENOUGH EVIDENCE",
    "NOT ENOUGH INFO": "NOT ENOUGH EVIDENCE",
    "CONFLICTING": "CONFLICTING",
    "CONFLICTING EVIDENCE": "CONFLICTING",
    "MIXED": "CONFLICTING",
    "BOTH": "CONFLICTING",
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a dictionary: {path}")
    return cfg


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    if path_obj.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object on line {line_no}, got {type(obj)}")
                rows.append(obj)
        return rows

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not all(isinstance(x, dict) for x in data):
            raise ValueError("JSON list must contain objects/dicts only.")
        return data

    # Optional support for {"data": [...]} style files.
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        rows = data["data"]
        if not all(isinstance(x, dict) for x in rows):
            raise ValueError("JSON data list must contain objects/dicts only.")
        return rows

    raise ValueError("Test file must be a JSON list, JSONL file, or {'data': [...]} object.")


def find_adapter_dir(output_dir: str) -> str:
    """
    LLaMA-Factory may save adapter files directly in output_dir or inside
    checkpoint-* subfolders. This picks the latest checkpoint containing
    adapter_config.json.
    """
    root = Path(output_dir)

    if (root / "adapter_config.json").exists():
        return str(root)

    checkpoints: List[Tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        if not (path / "adapter_config.json").exists():
            continue
        match = re.search(r"checkpoint-(\d+)", path.name)
        step = int(match.group(1)) if match else -1
        checkpoints.append((step, path))

    if not checkpoints:
        raise FileNotFoundError(
            "No LoRA/QLoRA adapter_config.json found in "
            f"{root} or in {root}/checkpoint-* folders."
        )

    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def clean_label_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().upper()
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("VERDICT:", " ")
    cleaned = cleaned.replace("LABEL:", " ")
    cleaned = re.sub(r"[^A-Z ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_label(text: Optional[str]) -> str:
    """Normalize model/gold text into one PAFC label or INVALID."""
    cleaned = clean_label_text(text)
    if not cleaned:
        return "INVALID"

    if cleaned in LABEL_ALIASES:
        return LABEL_ALIASES[cleaned]

    # Prefer longer labels first so "NOT ENOUGH EVIDENCE" wins before partial words.
    for key in sorted(LABEL_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", cleaned):
            return LABEL_ALIASES[key]

    return "INVALID"


def extract_gold(sample: Dict[str, Any]) -> str:
    """Get gold label from Alpaca-style output or ShareGPT-style assistant message."""
    if "output" in sample:
        return normalize_label(sample.get("output"))

    messages = sample.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return normalize_label(msg.get("content"))

    return "INVALID"


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build chat messages for inference.

    IMPORTANT:
    - Uses system + instruction + input only.
    - Does NOT pass sample['output'] or assistant gold label to the model.
    """
    # Standard Alpaca/LLaMA-Factory format.
    if "instruction" in sample or "input" in sample or "system" in sample:
        system_prompt = str(sample.get("system", "") or "").strip()
        instruction = str(sample.get("instruction", "") or "").strip()
        input_text = str(sample.get("input", "") or "").strip()

        user_content = instruction
        if input_text:
            user_content = f"{user_content}\n\n{input_text}" if user_content else input_text

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    # Optional ShareGPT/messages support.
    raw_messages = sample.get("messages")
    if isinstance(raw_messages, list):
        messages = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue
            if role == "assistant":
                continue  # Never leak the gold label during inference.
            if role in {"system", "user"}:
                messages.append({"role": role, "content": content})
        if messages:
            return messages

    raise ValueError("Sample does not contain instruction/input/system or messages fields.")


def choose_dtype(cfg: Dict[str, Any], dtype_arg: str) -> torch.dtype:
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "fp32":
        return torch.float32

    # auto: follow YAML bf16 when possible, otherwise fp16 on CUDA.
    wants_bf16 = bool(cfg.get("bf16", False))
    if wants_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(base_model_name_or_path: str, adapter_dir: str, use_fast: bool) -> AutoTokenizer:
    # Prefer adapter tokenizer if LLaMA-Factory saved it; otherwise use base tokenizer.
    tokenizer_path = adapter_dir if (Path(adapter_dir) / "tokenizer_config.json").exists() else base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_merged_model(
    base_model_name_or_path: str,
    adapter_dir: str,
    dtype: torch.dtype,
    attn_implementation: Optional[str] = None,
    safe_merge: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Base model     : {base_model_name_or_path}")
    print(f"QLoRA adapter  : {adapter_dir}")
    print(f"Merge dtype    : {dtype}")

    tokenizer = load_tokenizer(
        base_model_name_or_path=base_model_name_or_path,
        adapter_dir=adapter_dir,
        use_fast=True,
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    print("Loading base model in full precision for merge, not in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        **model_kwargs,
    )

    print("Loading QLoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )

    print("Merging adapter into base model with merge_and_unload()...")
    model = peft_model.merge_and_unload(safe_merge=safe_merge)
    model.eval()

    return model, tokenizer


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sample: Dict[str, Any],
    max_new_tokens: int,
    max_input_length: int,
) -> Tuple[str, str]:
    messages = build_messages(sample)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][prompt_len:]
    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return normalize_label(raw_output), raw_output


def save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions(
    path: Path,
    data: Iterable[Dict[str, Any]],
    y_true: List[str],
    y_pred: List[str],
    raw_outputs: List[str],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx, (sample, gold, pred, raw) in enumerate(zip(data, y_true, y_pred, raw_outputs)):
            row = {
                "id": sample.get("id", idx),
                "gold": gold,
                "prediction": pred,
                "raw_model_output": raw,
                "correct": gold == pred,
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "system": sample.get("system", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PAFC Top-10 QLoRA adapter into base model and evaluate."
    )
    parser.add_argument("--config", required=True, help="LLaMA-Factory QLoRA YAML config.")
    parser.add_argument("--test_file", default="data/pafc_top10_eval.json", help="PAFC eval JSON/JSONL file.")
    parser.add_argument("--adapter_dir", default=None, help="Explicit QLoRA adapter dir. Default: output_dir from YAML/latest checkpoint.")
    parser.add_argument("--out_dir", default="eval_outputs/pafc_qlora_top10", help="Directory to save metrics and predictions.")
    parser.add_argument("--save_merged_dir", default=None, help="Optional directory to save merged model and tokenizer.")
    parser.add_argument("--sample_index", type=int, default=0, help="Print this sample before full evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=8, help="Max tokens for label generation.")
    parser.add_argument("--max_input_length", type=int, default=None, help="Prompt truncation length. Default: YAML cutoff_len or 4096.")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="Dtype for full-precision merge.")
    parser.add_argument("--attn_implementation", default=None, choices=[None, "eager", "sdpa", "flash_attention_2"], help="Optional transformers attention implementation.")
    parser.add_argument("--safe_merge", action="store_true", help="Use PEFT safe_merge=True.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick debugging.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_model_name_or_path = cfg["model_name_or_path"]
    adapter_dir = args.adapter_dir or find_adapter_dir(cfg["output_dir"])
    max_input_length = args.max_input_length or int(cfg.get("cutoff_len", 4096))
    dtype = choose_dtype(cfg, args.dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PAFC Top-10 QLoRA merge + inference")
    print(f"Config         : {args.config}")
    print(f"Test file      : {args.test_file}")
    print(f"Output dir     : {out_dir}")
    print(f"Max input len  : {max_input_length}")
    print("=" * 80)

    data = load_json_or_jsonl(args.test_file)
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError("No test samples loaded.")

    model, tokenizer = load_merged_model(
        base_model_name_or_path=base_model_name_or_path,
        adapter_dir=adapter_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        safe_merge=args.safe_merge,
    )

    if args.save_merged_dir:
        save_dir = Path(args.save_merged_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving merged model to: {save_dir}")
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)

    # Print one debug sample first.
    sample_index = min(max(args.sample_index, 0), len(data) - 1)
    sample = data[sample_index]
    debug_pred, debug_raw = generate_one(
        model=model,
        tokenizer=tokenizer,
        sample=sample,
        max_new_tokens=args.max_new_tokens,
        max_input_length=max_input_length,
    )
    debug_gold = extract_gold(sample)

    print("\n" + "=" * 80)
    print(f"Single sample index: {sample_index}")
    print(f"Gold label     : {debug_gold}")
    print(f"Predicted label: {debug_pred}")
    print(f"Raw model text : {repr(debug_raw)}")
    print("=" * 80 + "\n")

    y_true: List[str] = []
    y_pred: List[str] = []
    raw_outputs: List[str] = []

    for sample in tqdm(data, desc="Evaluating"):
        gold = extract_gold(sample)
        pred, raw = generate_one(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            max_new_tokens=args.max_new_tokens,
            max_input_length=max_input_length,
        )
        y_true.append(gold)
        y_pred.append(pred)
        raw_outputs.append(raw)

    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        digits=4,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        digits=4,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    invalid_gold = sum(1 for x in y_true if x == "INVALID")
    invalid_pred = sum(1 for x in y_pred if x == "INVALID")

    print("\nAccuracy:", accuracy)
    print("\nClassification report:")
    print(report_text)
    print("\nConfusion matrix:")
    print("Labels:", LABELS)
    print(cm)
    print(f"\nInvalid gold labels: {invalid_gold}")
    print(f"Invalid predictions: {invalid_pred}")

    predictions_path = out_dir / "predictions.jsonl"
    report_path = out_dir / "classification_report.txt"
    metrics_path = out_dir / "metrics.json"
    cm_path = out_dir / "confusion_matrix.json"

    save_predictions(predictions_path, data, y_true, y_pred, raw_outputs)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    save_json(metrics_path, {
        "config": args.config,
        "test_file": args.test_file,
        "base_model_name_or_path": base_model_name_or_path,
        "adapter_dir": adapter_dir,
        "num_samples": len(data),
        "labels": LABELS,
        "accuracy": accuracy,
        "invalid_gold_labels": invalid_gold,
        "invalid_predictions": invalid_pred,
        "classification_report": report_dict,
    })
    save_json(cm_path, {
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
    })

    print("\nSaved files:")
    print(f"- {predictions_path}")
    print(f"- {report_path}")
    print(f"- {metrics_path}")
    print(f"- {cm_path}")

    # Help release memory when running multiple configs in one shell session.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
