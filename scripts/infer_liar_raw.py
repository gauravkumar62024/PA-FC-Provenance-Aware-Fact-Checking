import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from llmzoo.datasets.datasets import SYSTEM_PROMPTS


LABELS = ["TRUE", "HALF-TRUE", "FALSE"]
IGNORE_INDEX = -100

LABEL_PATTERN = re.compile(
    r"\b(HALF-TRUE|TRUE|FALSE)\b",
    flags=re.IGNORECASE,
)


def load_json_or_jsonl(path: str) -> List[Dict]:
    input_path = Path(path)

    with input_path.open("r", encoding="utf-8") as file:
        first_character = file.read(1)
        file.seek(0)

        if first_character == "[":
            data = json.load(file)
        else:
            data = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples in {path}")

    return data


def extract_user_text(example: Dict) -> str:
    for turn in example.get("conversations", []):
        role = str(turn.get("from", "")).strip().lower()

        if role in {"human", "user"}:
            text = str(turn.get("value", "")).strip()

            if text:
                return text

    # Supports alternative prepared files.
    for key in ["instruction", "input", "prompt"]:
        text = str(example.get(key, "")).strip()
        if text:
            return text

    raise ValueError(
        f"No user input found for sample {example.get('id', 'unknown')}"
    )


def extract_gold_label(example: Dict) -> str:
    for turn in example.get("conversations", []):
        role = str(turn.get("from", "")).strip().lower()

        if role in {"gpt", "assistant"}:
            value = str(turn.get("value", ""))
            match = LABEL_PATTERN.search(value)

            if match:
                return match.group(1).upper()

    for key in ["label", "gold_label", "output", "target"]:
        value = str(example.get(key, ""))
        match = LABEL_PATTERN.search(value)

        if match:
            return match.group(1).upper()

    raise ValueError(
        f"No gold label found for sample {example.get('id', 'unknown')}"
    )


def truncate_prompt(
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """
    Uses the same head-plus-tail truncation policy as training.

    It preserves:
    - system instruction and beginning of the claim/evidence;
    - ending chat-template tokens;
    - the complete candidate verdict whenever possible.
    """
    prompt_ids = list(prompt_ids)
    target_ids = list(target_ids)

    if len(target_ids) >= max_length:
        return [], target_ids[:max_length]

    prompt_budget = max_length - len(target_ids)

    if len(prompt_ids) <= prompt_budget:
        return prompt_ids, target_ids

    tail_keep = min(
        64,
        max(1, prompt_budget // 8),
        prompt_budget,
    )

    head_keep = prompt_budget - tail_keep

    prompt_ids = (
        prompt_ids[:head_keep]
        + prompt_ids[-tail_keep:]
    )

    return prompt_ids, target_ids


def construct_candidate(
    tokenizer,
    user_text: str,
    label: str,
    max_length: int,
) -> Tuple[List[int], int, int]:
    prompt_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPTS["liar_raw_cev"],
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    complete_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": f"Verdict: {label}",
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    complete_text = tokenizer.apply_chat_template(
        complete_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if not complete_text.startswith(prompt_text):
        raise RuntimeError(
            "The complete chat does not start with the prompt. "
            "Check the Qwen chat template."
        )

    target_text = complete_text[len(prompt_text):]

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    prompt_ids, target_ids = truncate_prompt(
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        max_length=max_length,
    )

    input_ids = prompt_ids + target_ids
    target_start = len(prompt_ids)
    target_length = len(target_ids)

    return input_ids, target_start, target_length


@torch.inference_mode()
def predict_one(
    model,
    tokenizer,
    user_text: str,
    max_length: int,
) -> Tuple[str, Dict[str, float], int]:
    candidate_records = [
        construct_candidate(
            tokenizer=tokenizer,
            user_text=user_text,
            label=label,
            max_length=max_length,
        )
        for label in LABELS
    ]

    maximum_candidate_length = max(
        len(record[0])
        for record in candidate_records
    )

    padded_input_ids = []
    attention_masks = []

    for input_ids, _, _ in candidate_records:
        padding_length = maximum_candidate_length - len(input_ids)

        padded_input_ids.append(
            input_ids
            + [tokenizer.pad_token_id] * padding_length
        )

        attention_masks.append(
            [1] * len(input_ids)
            + [0] * padding_length
        )

    device = next(model.parameters()).device

    input_tensor = torch.tensor(
        padded_input_ids,
        dtype=torch.long,
        device=device,
    )

    attention_tensor = torch.tensor(
        attention_masks,
        dtype=torch.long,
        device=device,
    )

    output = model(
        input_ids=input_tensor,
        attention_mask=attention_tensor,
        use_cache=False,
    )

    log_probabilities = torch.log_softmax(
        output.logits.float(),
        dim=-1,
    )

    label_scores: Dict[str, float] = {}

    for candidate_index, label in enumerate(LABELS):
        input_ids, target_start, target_length = (
            candidate_records[candidate_index]
        )

        target_token_scores = []

        for token_position in range(
            target_start,
            target_start + target_length,
        ):
            # Token at position t is predicted by logits at t-1.
            if token_position == 0:
                continue

            target_token_id = input_ids[token_position]

            token_score = log_probabilities[
                candidate_index,
                token_position - 1,
                target_token_id,
            ]

            target_token_scores.append(token_score)

        if not target_token_scores:
            raise RuntimeError(
                f"No target tokens were scored for label {label}"
            )

        # Mean normalization avoids penalizing HALF-TRUE because
        # it may use more tokens than TRUE or FALSE.
        mean_log_probability = torch.stack(
            target_token_scores
        ).mean()

        label_scores[label] = float(
            mean_log_probability.item()
        )

    predicted_label = max(
        label_scores,
        key=label_scores.get,
    )

    return (
        predicted_label,
        label_scores,
        maximum_candidate_length,
    )


def load_existing_predictions(path: Path) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}

    if not path.exists():
        return existing

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            sample_id = str(row["id"])

            if sample_id in existing:
                raise ValueError(
                    f"Duplicate prediction ID in existing file: {sample_id}"
                )

            existing[sample_id] = row

    return existing


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
    )
    parser.add_argument(
        "--test_file",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=1251,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from predictions.jsonl if inference was interrupted.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    test_data = load_json_or_jsonl(args.test_file)

    print("Test file:", args.test_file)
    print("Loaded test samples:", len(test_data))

    if len(test_data) != args.expected_count:
        raise ValueError(
            f"Expected exactly {args.expected_count} test samples, "
            f"but loaded {len(test_data)}."
        )

    sample_ids = [
        str(
            example.get(
                "id",
                example.get("claim_id", index),
            )
        )
        for index, example in enumerate(test_data)
    ]

    if len(sample_ids) != len(set(sample_ids)):
        duplicate_ids = [
            sample_id
            for sample_id, count
            in Counter(sample_ids).items()
            if count > 1
        ]

        raise ValueError(
            f"Duplicate test IDs found: {duplicate_ids[:10]}"
        )

    existing_predictions = {}

    if args.resume:
        existing_predictions = load_existing_predictions(
            prediction_path
        )
        print(
            "Existing predictions available for resume:",
            len(existing_predictions),
        )
    elif prediction_path.exists():
        raise FileExistsError(
            f"{prediction_path} already exists. "
            "Use a new output directory or pass --resume."
        )

    print("Loading tokenizer:", args.adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        use_fast=True,
        trust_remote_code=True,
    )

    tokenizer.model_max_length = args.max_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model:", args.base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("Loading LoRA adapter:", args.adapter_path)

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
    )

    model.eval()
    model.config.use_cache = False

    write_mode = "a" if args.resume else "w"
    start_time = time.time()

    with prediction_path.open(
        write_mode,
        encoding="utf-8",
    ) as writer:
        for index, example in enumerate(
            tqdm(
                test_data,
                total=len(test_data),
                desc="REFLEX test inference",
            )
        ):
            sample_id = sample_ids[index]

            if sample_id in existing_predictions:
                continue

            user_text = extract_user_text(example)
            gold_label = extract_gold_label(example)

            predicted_label, label_scores, input_length = (
                predict_one(
                    model=model,
                    tokenizer=tokenizer,
                    user_text=user_text,
                    max_length=args.max_length,
                )
            )

            row = {
                "id": sample_id,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "label_scores": label_scores,
                "input_length": input_length,
            }

            writer.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            writer.flush()

    runtime_seconds = time.time() - start_time

    all_predictions = load_existing_predictions(
        prediction_path
    )

    if len(all_predictions) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} predictions, "
            f"but prediction file contains {len(all_predictions)}."
        )

    ordered_rows = [
        all_predictions[sample_id]
        for sample_id in sample_ids
    ]

    y_true = [
        row["gold_label"]
        for row in ordered_rows
    ]

    y_pred = [
        row["predicted_label"]
        for row in ordered_rows
    ]

    invalid_predictions = [
        prediction
        for prediction in y_pred
        if prediction not in LABELS
    ]

    if invalid_predictions:
        raise RuntimeError(
            "Invalid predictions were produced: "
            f"{invalid_predictions[:10]}"
        )

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    metrics = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "test_file": args.test_file,
        "max_length": args.max_length,
        "num_test_samples": len(test_data),
        "num_predictions": len(y_pred),
        "num_filtered": 0,
        "num_invalid_predictions": 0,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "gold_label_counts": dict(Counter(y_true)),
        "predicted_label_counts": dict(Counter(y_pred)),
        "label_order": LABELS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "runtime_seconds_current_run": runtime_seconds,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n==========================================")
    print("REFLEX LIAR-RAW full-test evaluation")
    print("==========================================")
    print(f"Test samples:        {len(test_data)}")
    print(f"Predictions:         {len(y_pred)}")
    print("Filtered samples:    0")
    print("Invalid predictions: 0")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Macro-F1:            {macro_f1:.4f}")
    print(f"Weighted-F1:         {weighted_f1:.4f}")
    print(f"Macro-Precision:     {macro_precision:.4f}")
    print(f"Macro-Recall:        {macro_recall:.4f}")
    print(f"Gold counts:         {dict(Counter(y_true))}")
    print(f"Prediction counts:   {dict(Counter(y_pred))}")
    print("Confusion matrix label order:", LABELS)
    print(matrix)
    print("Predictions file:", prediction_path)
    print("Metrics file:", metrics_path)


if __name__ == "__main__":
    main()
