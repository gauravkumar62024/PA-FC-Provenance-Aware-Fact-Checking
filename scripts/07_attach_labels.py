#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()

    # JSON list / dict
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ["data", "records", "examples", "claims"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]
    except json.JSONDecodeError:
        pass

    # JSONL
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_claim(text: str) -> str:
    """
    Strong but safe normalization for claim matching.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def normalize_label(label: str) -> str:
    """
    Convert original AVeriTeC labels into the 4 training labels.
    Original AVeriTeC labels can be:
      Supported
      Refuted
      Not Enough Evidence
      Conflicting Evidence/Cherrypicking
    """
    x = str(label).strip().upper()

    mapping = {
        # Supported
        "SUPPORTED": "SUPPORTED",
        "SUPPORT": "SUPPORTED",
        "SUPPORTS": "SUPPORTED",

        # Refuted
        "REFUTED": "REFUTED",
        "REFUTE": "REFUTED",
        "REFUTES": "REFUTED",

        # NEI
        "NOT ENOUGH EVIDENCE": "NOT ENOUGH EVIDENCE",
        "NOT ENOUGH INFO": "NOT ENOUGH EVIDENCE",
        "NOT ENOUGH INFORMATION": "NOT ENOUGH EVIDENCE",
        "NEI": "NOT ENOUGH EVIDENCE",

        # Conflicting
        "CONFLICTING": "CONFLICTING",
        "CONFLICT": "CONFLICTING",
        "CONFLICTED": "CONFLICTING",
        "CONFLICTING EVIDENCE/CHERRYPICKING": "CONFLICTING",
        "CONFLICTING EVIDENCE / CHERRYPICKING": "CONFLICTING",
        "CHERRYPICKING": "CONFLICTING",
        "CHERRY-PICKING": "CONFLICTING",
        "CHERRY PICKING": "CONFLICTING",
    }

    if x not in mapping:
        raise ValueError(f"Unknown label found: {label}")

    return mapping[x]


def extract_claim_from_input_text(input_text: str) -> Optional[str]:
    """
    Extract claim from converted input_text like:

    CLAIM_ID: 0

    CLAIM:
    Some claim here.

    EVIDENCE_QA_1:
    ...
    """
    if not input_text:
        return None

    marker = "CLAIM:"
    if marker not in input_text:
        return None

    after = input_text.split(marker, 1)[1]

    # Stop before first evidence block if present
    stop_markers = [
        "\nEVIDENCE_QA_1:",
        "\nEVIDENCE QA 1:",
        "\nF1_NLI_1:",
    ]

    claim_part = after
    for sm in stop_markers:
        if sm in claim_part:
            claim_part = claim_part.split(sm, 1)[0]

    claim_part = claim_part.strip()
    return claim_part if claim_part else None


def get_converted_claim(row: Dict[str, Any]) -> Optional[str]:
    """
    Converted row may have claim directly, or only inside input_text.
    """
    if row.get("claim"):
        return str(row["claim"]).strip()

    input_text = row.get("input_text", "")
    return extract_claim_from_input_text(input_text)


def build_claim_to_label(original_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claim_to_label = {}
    duplicates = {}

    for i, row in enumerate(original_rows):
        claim = row.get("claim")
        label = row.get("label")

        if claim is None:
            raise KeyError(f"Original row {i} has no claim key.")
        if label is None:
            raise KeyError(f"Original row {i} has no label key.")

        norm_claim = normalize_claim(claim)
        norm_label = normalize_label(label)

        if norm_claim in claim_to_label and claim_to_label[norm_claim] != norm_label:
            duplicates.setdefault(norm_claim, set()).update(
                [claim_to_label[norm_claim], norm_label]
            )

        claim_to_label[norm_claim] = norm_label

    if duplicates:
        print("WARNING: Found duplicate claims with different labels:")
        for claim, labels in list(duplicates.items())[:10]:
            print(f"  labels={labels} claim={claim[:120]}")

    return claim_to_label


def add_labels(
    converted_path: str,
    original_path: str,
    output_path: str,
    unmatched_path: Optional[str] = None,
):
    converted_rows = load_json_or_jsonl(converted_path)
    original_rows = load_json_or_jsonl(original_path)

    claim_to_label = build_claim_to_label(original_rows)

    matched = 0
    unmatched = []

    for i, row in enumerate(converted_rows):
        claim = get_converted_claim(row)

        if claim is None:
            row["label"] = None
            row["label_match_status"] = "missing_claim_in_converted"
            unmatched.append(row)
            continue

        norm_claim = normalize_claim(claim)

        if norm_claim in claim_to_label:
            row["label"] = claim_to_label[norm_claim]
            row["label_match_status"] = "matched_by_claim"
            matched += 1
        else:
            row["label"] = None
            row["label_match_status"] = "unmatched_by_claim"
            row["extracted_claim_for_debug"] = claim
            unmatched.append(row)

    write_jsonl(converted_rows, output_path)

    if unmatched_path:
        write_jsonl(unmatched, unmatched_path)

    print("Done.")
    print(f"Converted rows: {len(converted_rows)}")
    print(f"Original rows: {len(original_rows)}")
    print(f"Matched labels: {matched}")
    print(f"Unmatched labels: {len(unmatched)}")
    print(f"Saved labeled file: {output_path}")

    if unmatched_path:
        print(f"Saved unmatched debug file: {unmatched_path}")

    if unmatched:
        print("\nFirst few unmatched claims:")
        for row in unmatched[:5]:
            claim = get_converted_claim(row)
            print("-", str(claim)[:200])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted", required=True, help="Converted JSONL file, e.g. train_format_top3.jsonl")
    parser.add_argument("--original", required=True, help="Original AVeriTeC JSON file with claim and label")
    parser.add_argument("--output", required=True, help="Output labeled JSONL file")
    parser.add_argument("--unmatched_output", default=None, help="Optional JSONL file for unmatched rows")

    args = parser.parse_args()

    add_labels(
        converted_path=args.converted,
        original_path=args.original,
        output_path=args.output,
        unmatched_path=args.unmatched_output,
    )


if __name__ == "__main__":
    main()
