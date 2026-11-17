import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


LABELS = ["TRUE", "FALSE"]


def norm_label(x):
    x = str(x).strip().upper()
    if x.startswith("TRUE"):
        return "TRUE"
    if x.startswith("FALSE"):
        return "FALSE"
    if "TRUE" in x and "FALSE" not in x:
        return "TRUE"
    if "FALSE" in x and "TRUE" not in x:
        return "FALSE"
    return "UNKNOWN"


def extract_label(text):
    text = str(text).strip()
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    # Prefer explicit first occurrence.
    m = re.search(r"\b(TRUE|FALSE)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return norm_label(text)


def build_messages(ex):
    system = ex.get("system", "")
    instruction = ex.get("instruction", "")
    inp = ex.get("input", "")

    user = instruction.strip()
    if inp.strip():
        user = user + "\n\n" + inp.strip()

    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user.strip()})
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    test_path = Path(args.test_file)
    data = json.loads(test_path.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    print("Loaded test examples:", len(data))

    # Use adapter tokenizer because LLaMAFactory saved chat template/tokenizer files there.
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    y_true = []
    y_pred = []
    rows = []

    for idx, ex in enumerate(tqdm(data, desc="Inference")):
        gold = norm_label(ex.get("output", ""))
        messages = build_messages(ex)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw_pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred = extract_label(raw_pred)

        y_true.append(gold)
        y_pred.append(pred)

        rows.append({
            "idx": idx,
            "gold": gold,
            "pred": pred,
            "raw_pred": raw_pred,
            "correct": gold == pred,
            "input_preview": ex.get("input", "")[:500],
        })

    valid_true = []
    valid_pred = []
    unknown = 0

    for g, p in zip(y_true, y_pred):
        if p not in LABELS:
            unknown += 1
            # Count unknown as wrong by assigning opposite label.
            p = "FALSE" if g == "TRUE" else "TRUE"
        valid_true.append(g)
        valid_pred.append(p)

    metrics = {
        "num_examples": len(data),
        "unknown_predictions": unknown,
        "accuracy": accuracy_score(valid_true, valid_pred),
        "macro_f1": f1_score(valid_true, valid_pred, labels=LABELS, average="macro"),
        "weighted_f1": f1_score(valid_true, valid_pred, labels=LABELS, average="weighted"),
        "classification_report": classification_report(
            valid_true,
            valid_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": confusion_matrix(valid_true, valid_pred, labels=LABELS).tolist(),
    }

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    Path(args.metrics_json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print("prediction file:", args.output_jsonl)
    print("metric file:", args.metrics_json)


if __name__ == "__main__":
    main()
