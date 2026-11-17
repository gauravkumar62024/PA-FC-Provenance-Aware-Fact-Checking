import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LABELS = ["entailment", "contradiction", "neutral", "mixture"]

def normalize_label(text):
    t = str(text).lower().strip()
    t = t.replace("</s>", "").replace("<pad>", "").strip()
    t = re.sub(r"[^a-z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    mapping = {
        "true": "entailment",
        "support": "entailment",
        "supports": "entailment",
        "supported": "entailment",
        "entailment": "entailment",
        "entailed": "entailment",

        "false": "contradiction",
        "refute": "contradiction",
        "refutes": "contradiction",
        "refuted": "contradiction",
        "contradiction": "contradiction",
        "contradict": "contradiction",

        "unproven": "neutral",
        "not enough evidence": "neutral",
        "not enough information": "neutral",
        "unknown": "neutral",
        "nei": "neutral",
        "neutral": "neutral",

        "mixture": "mixture",
        "mixed": "mixture",
        "conflicting": "mixture",
        "conflict": "mixture",
    }

    if t in mapping:
        return mapping[t]

    if "entailment" in t or "supported" in t or "support" in t or t == "true":
        return "entailment"
    if "contradiction" in t or "contradict" in t or "refuted" in t or "refute" in t or t == "false":
        return "contradiction"
    if "not enough evidence" in t or "not enough information" in t or "unproven" in t or "neutral" in t or "unknown" in t or "nei" in t:
        return "neutral"
    if "mixture" in t or "mixed" in t or "conflicting" in t or "conflict" in t:
        return "mixture"

    return t

def build_prompt(tok, ex):
    messages = [
        {"role": "system", "content": ex.get("system", "")},
        {"role": "user", "content": ex["instruction"] + "\n\n" + ex["input"]},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    base_model = cfg["model_name_or_path"]
    adapter = cfg["output_dir"]

    Path(args.pred_dir).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    data = json.load(open(args.test_file))
    gold, pred, raw_texts = [], [], []

    for i, ex in enumerate(data):
        prompt = build_prompt(tok, ex)
        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        y = normalize_label(ex["output"])
        p = normalize_label(gen)

        gold.append(y)
        pred.append(p)
        raw_texts.append(gen)

        if (i + 1) % 100 == 0:
            print("done", i + 1, "/", len(data))

    metrics = {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, pred, labels=LABELS, average="weighted", zero_division=0),
        "labels": LABELS,
        "gold_label_counts": pd.Series(gold).value_counts().to_dict(),
        "pred_label_counts": pd.Series(pred).value_counts().to_dict(),
        "classification_report": classification_report(
            gold, pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(gold, pred, labels=LABELS).tolist(),
    }

    with open(Path(args.pred_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "gold": gold,
        "pred": pred,
        "raw_generation": raw_texts,
    }).to_csv(Path(args.pred_dir) / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
