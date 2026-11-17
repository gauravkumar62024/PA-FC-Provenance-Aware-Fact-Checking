#!/usr/bin/env python3
"""
Convert PA-FC labeled data into LLaMA-Factory SFT data.

Supported input rows:
  1) Compact/verbose JSONL with: {"input_text": ..., "label": ...}
  2) Chat JSONL with: {"messages": [{"role":"system"}, {"role":"user"}, {"role":"assistant"}]}

Recommended pipeline:
  train/dev original top10 -> compact JSONL -> this script -> LLaMA-Factory data/*.json

Outputs:
  1) <output>                      Alpaca or ShareGPT JSON array
  2) <output>.dataset_info.json    snippet to merge into LLaMA-Factory data/dataset_info.json
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH EVIDENCE", "CONFLICTING"]

SYSTEM_PROMPT = """You are a fact verification system. You judge the veracity of a claim using retrieved evidence that has been annotated with provenance-aware quality features.

Each evidence block contains:
- Q / A: a question-answer pair generated from the evidence document.
- NLI: the relation of this evidence to the claim (support / refute / nei) with confidence.
- SRC: source type (fact_checker, est_news, gov, academic, aggregator, blog, forum, unknown, ...).
- TRUST: domain trust level (high, medium_high, medium, low, ...).
- PROV: provenance quality level (strong, moderate, weak).
- FLAGS: warnings such as leakage_sensitive (source may reveal the gold verdict), fact_check (the page is itself a fact-check article), satire, low_credibility, archive_copy, primary_source.

Weigh evidence by its quality: prefer high-trust, strong-provenance, independent sources. Be cautious with leakage_sensitive, satire, and low_credibility evidence. If evidence items genuinely contradict each other, the verdict is CONFLICTING. If the evidence is insufficient or irrelevant, the verdict is NOT ENOUGH EVIDENCE."""

INSTRUCTION_LABEL_ONLY = """Classify the veracity of the claim below using the annotated evidence.

Answer with exactly one label and nothing else:
SUPPORTED, REFUTED, NOT ENOUGH EVIDENCE, or CONFLICTING."""

INSTRUCTION_VERDICT_LINE = """Classify the veracity of the claim below using the annotated evidence.

Answer with exactly one line and nothing else:
VERDICT: SUPPORTED
VERDICT: REFUTED
VERDICT: NOT ENOUGH EVIDENCE
VERDICT: CONFLICTING"""

ASSISTANT_RE = re.compile(r"^\s*(?:VERDICT\s*:\s*)?(.*?)\s*$", re.IGNORECASE)
TASK_SPLIT_RE = re.compile(r"\n\s*Task\s*:\s*", re.IGNORECASE)


def normalize_label(label: Any) -> str:
    raw = str(label).strip()
    m = ASSISTANT_RE.match(raw)
    x = m.group(1).strip().upper() if m else raw.upper()
    # Handle accidental punctuation / one-line model outputs.
    x = re.sub(r"[.\s]+$", "", x)

    mapping = {
        "SUPPORT": "SUPPORTED",
        "SUPPORTED": "SUPPORTED",
        "SUPPORTS": "SUPPORTED",
        "REFUTE": "REFUTED",
        "REFUTED": "REFUTED",
        "REFUTES": "REFUTED",
        "NEI": "NOT ENOUGH EVIDENCE",
        "NOT ENOUGH INFO": "NOT ENOUGH EVIDENCE",
        "NOT ENOUGH INFORMATION": "NOT ENOUGH EVIDENCE",
        "NOT ENOUGH EVIDENCE": "NOT ENOUGH EVIDENCE",
        "CONFLICT": "CONFLICTING",
        "CONFLICTING": "CONFLICTING",
        "CONFLICTED": "CONFLICTING",
        "CONFLICTING EVIDENCE": "CONFLICTING",
    }
    if x not in mapping:
        raise ValueError(f"Unknown label: {label!r}")
    return mapping[x]


def get_message(messages: List[Dict[str, Any]], role: str) -> str:
    for msg in messages:
        if str(msg.get("role", "")).lower() == role:
            return str(msg.get("content", ""))
    return ""


def strip_task_from_user_text(text: str) -> str:
    """Remove the old chat-format Task block if present.

    The LLaMA-Factory example will use our own instruction field, so keeping the
    original Task block in the input would duplicate instructions.
    """
    return TASK_SPLIT_RE.split(text, maxsplit=1)[0].rstrip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if "input_text" in rec and "label" in rec:
                input_text = str(rec["input_text"]).strip()
                label = normalize_label(rec["label"])
            elif "messages" in rec:
                input_text = strip_task_from_user_text(get_message(rec["messages"], "user")).strip()
                label = normalize_label(get_message(rec["messages"], "assistant"))
            else:
                raise KeyError(
                    f"Line {line_no}: expected either input_text+label or messages."
                )

            if not input_text:
                raise ValueError(f"Line {line_no}: empty input text.")
            records.append({"input_text": input_text, "label": label})
    return records


def format_output_label(label: str, output_label_style: str) -> str:
    if output_label_style == "verdict_line":
        return f"VERDICT: {label}"
    return label


def to_alpaca(rec: Dict[str, str], instruction: str, output_label_style: str) -> Dict[str, str]:
    return {
        "instruction": instruction,
        "input": rec["input_text"],
        "output": format_output_label(rec["label"], output_label_style),
        "system": SYSTEM_PROMPT,
    }


def to_sharegpt(rec: Dict[str, str], instruction: str, output_label_style: str) -> Dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": instruction + "\n\n" + rec["input_text"]},
            {"from": "gpt", "value": format_output_label(rec["label"], output_label_style)},
        ],
        "system": SYSTEM_PROMPT,
    }


def oversample(records: List[Dict[str, str]], max_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, int]]]:
    """Duplicate minority classes up to min(majority_count, original_count * max_ratio).

    max_ratio <= 1 disables oversampling. Use this only for train, never eval/dev.
    """
    if max_ratio <= 1.0 or not records:
        return records, {}

    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec)

    majority = max(len(v) for v in by_label.values())
    out: List[Dict[str, str]] = []
    report: Dict[str, Dict[str, int]] = {}

    for label, items in by_label.items():
        target = min(majority, int(round(len(items) * max_ratio)))
        target = max(target, len(items))
        extra = []
        while len(items) + len(extra) < target:
            extra.append(rng.choice(items).copy())
        report[label] = {"original": len(items), "after": len(items) + len(extra)}
        out.extend(items)
        out.extend(extra)

    rng.shuffle(out)
    return out, report


def dataset_info_entry(dataset_name: str, file_name: str, style: str) -> Dict[str, Any]:
    if style == "alpaca":
        return {
            dataset_name: {
                "file_name": file_name,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                },
            }
        }
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            },
        }
    }


def write_json_array(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON path, e.g. data/pafc_train.json.")
    parser.add_argument("--dataset-name", required=True, help="Name used in dataset_info.json and training YAML.")
    parser.add_argument("--style", choices=["alpaca", "sharegpt"], default="alpaca")
    parser.add_argument(
        "--output-label-style",
        choices=["label", "verdict_line"],
        default="label",
        help="Use 'label' for outputs like SUPPORTED; use 'verdict_line' for VERDICT: SUPPORTED.",
    )
    parser.add_argument(
        "--oversample-max-ratio",
        type=float,
        default=1.0,
        help="Example: 3.0 duplicates minority classes up to 3x, capped at majority size. Use only for train.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = read_jsonl(input_path)
    print(f"Loaded {len(records)} records from {input_path}")
    print("Label distribution before:", dict(Counter(r["label"] for r in records)))

    records, report = oversample(records, args.oversample_max_ratio, args.seed)
    if report:
        print("Oversampling report:")
        for label in LABELS:
            if label in report:
                info = report[label]
                print(f"  {label}: {info['original']} -> {info['after']}")
        print("Label distribution after:", dict(Counter(r["label"] for r in records)))

    instruction = INSTRUCTION_VERDICT_LINE if args.output_label_style == "verdict_line" else INSTRUCTION_LABEL_ONLY
    if args.style == "alpaca":
        data = [to_alpaca(r, instruction, args.output_label_style) for r in records]
    else:
        data = [to_sharegpt(r, instruction, args.output_label_style) for r in records]

    write_json_array(output_path, data)
    print(f"Wrote {len(data)} SFT examples -> {output_path}")

    info = dataset_info_entry(args.dataset_name, output_path.name, args.style)
    info_path = output_path.with_suffix(".dataset_info.json")
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset_info snippet -> {info_path}")
    print("Merge that snippet into LLaMA-Factory's data/dataset_info.json.")


if __name__ == "__main__":
    main()
