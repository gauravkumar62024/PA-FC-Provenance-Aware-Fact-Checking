import os
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_JOB_TYPE"] = "test"

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix

from load_custom_dataset import load_raw_data
from feature_conversion_methods import formatting, label_mapping_t5, canonical_dataset_name
from causal_utils import build_inference_prompt_ids

label2text = {0: "entailment", 1: "neutral", 2: "contradiction", 3: "mixture"}


def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_label(text):
    t = str(text).lower().strip()
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

    for key, val in mapping.items():
        if key in t:
            return val

    return t


def build_prompt(tokenizer, input_string):
    """Deprecated: kept only for reference.

    Truncating the rendered prompt from the right removes the trailing
    generation prompt, so the model is never asked to answer. Use
    causal_utils.build_inference_prompt_ids instead.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a fact verification model. Return exactly one label."},
            {"role": "user", "content": input_string},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dataset_name", default="pubhealth", type=str)
    parser.add_argument("--target_dataset_name", default="pubhealth", type=str)
    parser.add_argument("--data_split", default="test", type=str)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--adapter_path", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sample_seed", default=42, type=int)
    parser.add_argument("--test_bsz", default=1, type=int)
    parser.add_argument("--max_length", default=4096, type=int)
    parser.add_argument("--max_new_tokens", default=8, type=int)
    parser.add_argument("--explanation_sep", default=' "explanation: " ', type=str)
    parser.add_argument("--io_format", default="no_exp", type=str)
    parser.add_argument("--result_path", default="../result_causal/", type=str)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--explanation_source", default="original", type=str)
    parser.add_argument("--limit", default=-1, type=int,
                        help="only score the first N test items (smoke tests); -1 = all")
    args = parser.parse_args()

    set_seed(args.seed)
    set_other_seeds(args.seed)

    result_path = Path(args.result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_raw_data(args.target_dataset_name, split=args.data_split)

    if args.limit > 0:
        n = min(args.limit, len(data))
        data = data.select(range(n))
        print(f"[smoke] truncated test set to {n} samples")

    input_strings = []
    answer_strings = []
    for item in data:
        inp, ans = formatting(item, args.target_dataset_name, args.explanation_sep, args.io_format, args.explanation_source)
        input_strings.append(" ".join(inp.split()))
        answer_strings.append(" ".join(ans.split()))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    dataset_key = canonical_dataset_name(args.target_dataset_name)
    num_classes = len(label_mapping_t5[dataset_key])
    label_names = [label_mapping_t5[dataset_key][i] for i in range(num_classes)]

    print(f"Inference on {len(data)} samples...")
    gold_labels = []
    pred_labels = []
    raw_generations = []

    for i in tqdm(range(len(data)), desc="Inferencing"):
        # Truncate the body, never the trailing generation prompt.
        prompt_ids = build_inference_prompt_ids(
            tokenizer, input_strings[i], args.max_length, reserve=args.max_new_tokens
        )
        input_ids = torch.tensor([prompt_ids], device=model.device)
        attn = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        raw_generations.append(gen_text.strip())

        gold_l = label_mapping_t5[dataset_key][data[i]["label"]]
        pred_l = normalize_label(gen_text)

        gold_labels.append(gold_l)
        pred_labels.append(pred_l)

    print("Evaluating...")
    accuracy = sum(g == p for g, p in zip(gold_labels, pred_labels)) / len(gold_labels)
    report = classification_report(gold_labels, pred_labels, labels=label_names, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold_labels, pred_labels, labels=label_names)

    results = {
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "gold_label_counts": {l: gold_labels.count(l) for l in label_names},
        "pred_label_counts": {l: pred_labels.count(l) for l in label_names},
    }

    with open(result_path / "results_test.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(result_path / "report_test.json", "w") as f:
        json.dump(report, f, indent=2)

    analysis_file = result_path / "test_posthoc_analysis.txt"
    with open(analysis_file, "w", encoding="utf-8") as g:
        for i in range(len(data)):
            g.write(f"Hypothesis: {data[i]['hypothesis']}\n")
            g.write(f"Premise: {str(data[i]['premise'])[:500]}\n")
            g.write(f"Correct: {gold_labels[i]}\n")
            g.write(f"Predicted: {pred_labels[i]}\n")
            g.write(f"Raw generation: {raw_generations[i]}\n")
            g.write(f"Match: {gold_labels[i] == pred_labels[i]}\n\n")

    with open(result_path / "test_generation_batch.txt", "w", encoding="utf-8") as f:
        for gen in raw_generations:
            f.write(gen.replace("\n", " ") + "\n")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Weighted-F1: {results['weighted_f1']:.4f}")
    print(f"\nResults saved to {result_path}")
    print("Finished inference.")


if __name__ == "__main__":
    main()
