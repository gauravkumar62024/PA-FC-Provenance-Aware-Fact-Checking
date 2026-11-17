#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


LABEL_KEYS = [
    "label",
    "gold_label",
    "veracity",
    "verdict",
    "claim_label",
    "true_label",
]


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8").strip()

    # Try normal JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            # common wrappers
            for key in ["data", "records", "examples", "claims"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = d.get(key, default)
    if value is None:
        return default
    return value


def nested_get(d: Dict[str, Any], path: List[str], default: Any = "") -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    return cur


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def get_label(record: Dict[str, Any]) -> Optional[str]:
    for key in LABEL_KEYS:
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def get_evidence_list(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Supports your current format:
      record["top_10"] = [evidence1, evidence2, ...]
    Also supports fallback:
      record["evidence"] = [...]
    """
    evidences = record.get("top_10", None)
    if evidences is None:
        evidences = record.get("evidence", [])

    if not isinstance(evidences, list):
        return []

    # Use rank only for ordering, but do NOT include rank in the final text.
    evidences = sorted(
        evidences,
        key=lambda x: x.get("rank", 999999) if isinstance(x, dict) else 999999
    )
    return [e for e in evidences if isinstance(e, dict)]


def format_one_evidence(ev: Dict[str, Any], idx: int) -> str:
    """
    Uses QA instead of raw evidence sentence.
    Skips retrieval features: rank, retrieval_score, source_doc_idx.
    """

    # QA can be flattened or nested inside pafc.qa
    question = safe_get(ev, "question") or nested_get(ev, ["pafc", "qa", "question"])
    answer = safe_get(ev, "answer") or nested_get(ev, ["pafc", "qa", "answer"])

    # F1 flattened
    f1_relation = safe_get(ev, "f1_relation_label")
    f1_conf = safe_get(ev, "f1_confidence")
    f1_support = safe_get(ev, "f1_p_support")
    f1_refute = safe_get(ev, "f1_p_refute")
    f1_neutral = safe_get(ev, "f1_p_neutral")

    # F2 flattened or nested
    f2_source_type = safe_get(ev, "f2_source_type") or nested_get(ev, ["pafc", "f2", "source_type"])
    f2_match_conf = safe_get(ev, "f2_match_confidence") or nested_get(ev, ["pafc", "f2", "match_confidence"])
    f2_leakage = safe_get(ev, "f2_leakage_sensitive")
    if f2_leakage == "":
        f2_leakage = nested_get(ev, ["pafc", "f2", "leakage_sensitive"])

    # F3 flattened or nested
    f3_trust_score = safe_get(ev, "f3_trust_score") or nested_get(ev, ["pafc", "f3", "trust_score"])
    f3_trust_level = safe_get(ev, "f3_trust_level") or nested_get(ev, ["pafc", "f3", "trust_level"])

    # F4 flattened or nested
    f4_labels = safe_get(ev, "f4_provenance_labels")
    if f4_labels == "":
        f4_labels = nested_get(ev, ["pafc", "f4", "provenance_labels"])

    f4_score = safe_get(ev, "f4_provenance_score") or nested_get(ev, ["pafc", "f4", "provenance_score"])
    f4_level = safe_get(ev, "f4_provenance_level") or nested_get(ev, ["pafc", "f4", "provenance_level"])

    f4_is_archive = safe_get(ev, "f4_is_archive_or_copy")
    if f4_is_archive == "":
        f4_is_archive = nested_get(ev, ["pafc", "f4", "is_archive_or_copy"])

    f4_is_fact_check = safe_get(ev, "f4_is_fact_check_article")
    if f4_is_fact_check == "":
        f4_is_fact_check = nested_get(ev, ["pafc", "f4", "is_fact_check_article"])

    f4_is_primary = safe_get(ev, "f4_is_primary_or_authoritative")
    if f4_is_primary == "":
        f4_is_primary = nested_get(ev, ["pafc", "f4", "is_primary_or_authoritative"])

    f4_is_low_cred = safe_get(ev, "f4_is_low_credibility_or_abuse")
    if f4_is_low_cred == "":
        f4_is_low_cred = nested_get(ev, ["pafc", "f4", "is_low_credibility_or_abuse"])

    f4_is_satire = safe_get(ev, "f4_is_satire_or_humor")
    if f4_is_satire == "":
        f4_is_satire = nested_get(ev, ["pafc", "f4", "is_satire_or_humor"])

    f4_is_reference = safe_get(ev, "f4_is_reference_or_data")
    if f4_is_reference == "":
        f4_is_reference = nested_get(ev, ["pafc", "f4", "is_reference_or_data"])

    f4_is_multimedia = safe_get(ev, "f4_is_multimedia")
    if f4_is_multimedia == "":
        f4_is_multimedia = nested_get(ev, ["pafc", "f4", "is_multimedia"])

    block = f"""
EVIDENCE_QA_{idx}:
question: {clean_value(question)}
answer: {clean_value(answer)}

F1_NLI_{idx}:
relation: {clean_value(f1_relation)}
confidence: {clean_value(f1_conf)}
support_probability: {clean_value(f1_support)}
refute_probability: {clean_value(f1_refute)}
neutral_probability: {clean_value(f1_neutral)}

F2_SOURCE_{idx}:
source_type: {clean_value(f2_source_type)}
match_confidence: {clean_value(f2_match_conf)}
leakage_sensitive: {clean_value(f2_leakage)}

F3_TRUST_{idx}:
trust_score: {clean_value(f3_trust_score)}
trust_level: {clean_value(f3_trust_level)}

F4_PROVENANCE_{idx}:
provenance_labels: {clean_value(f4_labels)}
provenance_score: {clean_value(f4_score)}
provenance_level: {clean_value(f4_level)}
is_archive_or_copy: {clean_value(f4_is_archive)}
is_fact_check_article: {clean_value(f4_is_fact_check)}
is_primary_or_authoritative: {clean_value(f4_is_primary)}
is_low_credibility_or_abuse: {clean_value(f4_is_low_cred)}
is_satire_or_humor: {clean_value(f4_is_satire)}
is_reference_or_data: {clean_value(f4_is_reference)}
is_multimedia: {clean_value(f4_is_multimedia)}
""".strip()

    return block


def build_input_text(record: Dict[str, Any], top_k: int) -> str:
    claim_id = safe_get(record, "claim_id")
    claim = safe_get(record, "claim")

    evidences = get_evidence_list(record)[:top_k]

    parts = [
        f"CLAIM_ID: {clean_value(claim_id)}",
        "",
        "CLAIM:",
        clean_value(claim),
    ]

    for i, ev in enumerate(evidences, start=1):
        parts.append("")
        parts.append(format_one_evidence(ev, i))

    return "\n".join(parts).strip()


def regex_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def load_tokenizers(model_names: List[str]):
    tokenizers = {}
    if not model_names:
        return tokenizers

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "WARNING: transformers is not installed. "
            "Only regex_token_count will be produced."
        )
        return tokenizers

    for model_name in model_names:
        try:
            tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"WARNING: could not load tokenizer {model_name}: {e}")

    return tokenizers


def tokenizer_count(tokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return len(encoded["input_ids"])


def convert(
    input_path: str,
    output_path: str,
    top_k: int,
    model_names: List[str],
    report_csv: Optional[str] = None,
):
    records = load_json_or_jsonl(input_path)
    tokenizers = load_tokenizers(model_names)

    report_rows = []

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            input_text = build_input_text(record, top_k=top_k)
            label = get_label(record)

            token_counts = {
                "regex_token_count": regex_token_count(input_text)
            }

            for model_name, tok in tokenizers.items():
                token_counts[model_name] = tokenizer_count(tok, input_text)

            output_obj = {
                "claim_id": record.get("claim_id", idx),
                "top_k": top_k,
                "input_text": input_text,
                "label": label,
                "token_counts": token_counts,
            }

            out_f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

            row = {
                "claim_id": record.get("claim_id", idx),
                "top_k": top_k,
                "label": label,
                **token_counts,
            }
            report_rows.append(row)

    if report_csv:
        all_keys = sorted({k for row in report_rows for k in row.keys()})
        with open(report_csv, "w", encoding="utf-8", newline="") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(report_rows)

    print(f"Saved converted data to: {output_path}")
    if report_csv:
        print(f"Saved token report to: {report_csv}")
    print(f"Total records: {len(records)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--top_k", type=int, default=1, help="Number of QA evidences to include")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["roberta-large", "xlnet-large-cased"],
        help="Tokenizer names for token counting",
    )
    parser.add_argument(
        "--report_csv",
        default=None,
        help="Optional CSV file for token count report",
    )

    args = parser.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        model_names=args.models,
        report_csv=args.report_csv,
    )


if __name__ == "__main__":
    main()
