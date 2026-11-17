#!/usr/bin/env python3
"""
Convert fact-check retrieval data + dev.json labels into LlamaFactory SFT JSONL.

This script uses the newer provenance-aware SYSTEM_PROMPT from
unslot_datafromte_conveter2.py and creates verdict-only SFT examples.

Output format, one JSON object per line:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Claim + metadata + evidence"},
    {"role": "assistant", "content": "VERDICT: REFUTED"}
  ]
}

This output can be registered in LlamaFactory as ShareGPT/OpenAI-style messages.
"""

from __future__ import annotations

import os as _os, sys as _sys
_PAFC_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _PAFC_SRC not in _sys.path:
    _sys.path.insert(0, _PAFC_SRC)

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a careful, skeptical, provenance-aware fact-checking verifier.

Given a claim and retrieved evidence items, silently decide exactly one veracity label.

Allowed outputs:
VERDICT: SUPPORTED
VERDICT: REFUTED
VERDICT: NOT ENOUGH EVIDENCE
VERDICT: CONFLICTING EVIDENCE

Use the evidence **answer** text as the primary source of factual information.

Use all evidence features when judging reliability and relevance:

* **f1 (Relation Analysis):** Indicates the predicted relationship between the evidence and the claim using a relation label and probabilities (`p_support`, `p_refute`, `p_neutral`). Use this to understand whether the evidence may support, refute, or be neutral toward the claim, but verify the relationship from the answer text itself.
* **f2 (Source Quality):** Includes `source_type` and `leakage_risk`. Use these to assess the nature of the source and whether the evidence may contain fact-check verdict leakage rather than independent factual support.
* **f3 (Trust Assessment):** Includes `trust_score`, `trust_level`, and trust reasons. Use these signals to estimate the overall credibility and reliability of the source.
* **f4 (Provenance & Reliability):** Includes `provenance_level`, `provenance_flags`, and indicators such as `primary_or_authoritative`, `fact_check_article`, `satire_or_humor`, and `low_credibility_or_abuse`. Use these to evaluate source origin, authenticity, authority, and potential reliability concerns.

Source weighting:

* High weight: official/government, academic, primary/authoritative, reputable independent sources, high trust scores, strong provenance, and low leakage risk.
* Low weight: forums, blogs, social media, unknown sources, satire/humor, low-credibility sources, and evidence with weak provenance.
* If `leakage_risk=YES` or `fact_check_article=YES`, ignore verdict-like language and use only factual statements that can independently support or refute the claim.

Decision rules:

* SUPPORTED: reliable evidence supports every essential part of the claim.
* REFUTED: reliable evidence contradicts any central part of the claim.
* NOT ENOUGH EVIDENCE: evidence is missing, weak, leaked, low-quality, or leaves an essential part unresolved.
* CONFLICTING EVIDENCE: reliable evidence supports both sides, or the claim is technically true but misleading through cherry-picking or omission of important context.

For causal, quote, numerical, time-sensitive, and absolute claims, verify the exact causal relationship, attribution, numbers, dates, units, scope, and wording.

Think internally only. Output exactly one line and nothing else:
VERDICT: <LABEL>
"""


LABEL_MAP = {
    "supported": "SUPPORTED",
    "support": "SUPPORTED",
    "true": "SUPPORTED",

    "refuted": "REFUTED",
    "refute": "REFUTED",
    "false": "REFUTED",

    "not enough evidence": "NOT ENOUGH EVIDENCE",
    "not enough info": "NOT ENOUGH EVIDENCE",
    "not_enough_evidence": "NOT ENOUGH EVIDENCE",
    "insufficient evidence": "NOT ENOUGH EVIDENCE",
    "nei": "NOT ENOUGH EVIDENCE",

    "conflicting evidence": "CONFLICTING EVIDENCE",
    "conflicting": "CONFLICTING EVIDENCE",
    "conflict": "CONFLICTING EVIDENCE",
    "cherrypicking": "CONFLICTING EVIDENCE",
    "cherry picking": "CONFLICTING EVIDENCE",
    "cherry-picking": "CONFLICTING EVIDENCE",
    "conflicting evidence cherrypicking": "CONFLICTING EVIDENCE",
    "conflicting evidence cherry picking": "CONFLICTING EVIDENCE",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_key(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(label: Any) -> str:
    raw = normalize_text(label)

    key = raw.lower()
    key = key.replace("/", " ")
    key = key.replace("-", " ")
    key = key.replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip()

    if key in LABEL_MAP:
        return LABEL_MAP[key]

    if "conflicting" in key or "cherrypicking" in key or "cherry picking" in key:
        return "CONFLICTING EVIDENCE"

    raise ValueError(f"Unknown label: {label!r} normalized as {key!r}")


def truncate_text(value: Any, max_chars: int) -> str:
    text = normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + " ... [truncated]"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if value is None:
        return "UNKNOWN"

    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "YES"
    if text in {"no", "false", "0", "n"}:
        return "NO"
    return "UNKNOWN"


def compact_list(value: Any, max_items: int = 8) -> str:
    if value is None or value == "":
        return "none"

    if isinstance(value, list):
        items = [normalize_text(x) for x in value if normalize_text(x)]
    else:
        items = [normalize_text(value)]

    if not items:
        return "none"

    shown = items[:max_items]
    extra = len(items) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text += f", ... +{extra} more"
    return text


def get_nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_first(data: dict[str, Any], keys_or_paths: list[Any], default: Any = None) -> Any:
    for item in keys_or_paths:
        if isinstance(item, list):
            value = get_nested(data, item, default=None)
        else:
            value = data.get(item)

        if value is not None and value != "":
            return value

    return default


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise ValueError(f"Unsupported JSON root type in {path}: {type(data)}")

        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValueError(f"Expected object at record {i} in {path}")

        return records

    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSONL at {path}:{line_no}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")

            records.append(obj)

        return records


def build_evidence_item(ev: dict[str, Any], idx: int, max_answer_chars: int) -> str:
    question = get_first(ev, ["question", ["qa", "question"]], default="")
    answer = get_first(ev, ["answer", ["qa", "answer"], "sentence"], default="")

    domain = get_first(
        ev,
        ["domain", "f3_domain", "f4_domain", "original_domain"],
        default="unknown",
    )
    url = get_first(
        ev,
        ["url", "qa_url", "retrieved_url", "canonical_source_url", ["qa", "url"]],
        default="unknown",
    )

    f1_label = get_first(ev, ["f1_relation_label"], default="nei")
    p_support = safe_float(get_first(ev, ["f1_p_support"], default=0.0))
    p_refute = safe_float(get_first(ev, ["f1_p_refute"], default=0.0))
    p_neutral = safe_float(get_first(ev, ["f1_p_neutral"], default=0.0))

    f2_source_type = get_first(
        ev,
        ["f2_source_type", ["pafc", "f2", "source_type"]],
        default="unknown",
    )
    f2_leakage_sensitive = get_first(
        ev,
        ["f2_leakage_sensitive", ["pafc", "f2", "leakage_sensitive"]],
        default=None,
    )

    f3_trust_score = safe_float(
        get_first(ev, ["f3_trust_score", ["pafc", "f3", "trust_score"]], default=0.0)
    )
    f3_trust_level = get_first(
        ev,
        ["f3_trust_level", ["pafc", "f3", "trust_level"]],
        default="unknown",
    )
    f3_reasons = get_first(
        ev,
        ["f3_trust_reasons", ["pafc", "f3", "reasons"]],
        default=[],
    )

    f4_provenance_level = get_first(
        ev,
        ["f4_provenance_level", ["pafc", "f4", "provenance_level"]],
        default="unknown",
    )
    f4_provenance_flags = get_first(
        ev,
        ["f4_provenance_labels", ["pafc", "f4", "provenance_labels"]],
        default=[],
    )
    f4_provenance_reasons = get_first(
        ev,
        ["f4_provenance_reasons", ["pafc", "f4", "provenance_reasons"]],
        default=[],
    )

    f4_fact_check_article = get_first(
        ev,
        ["f4_is_fact_check_article", ["pafc", "f4", "is_fact_check_article"]],
        default=None,
    )
    f4_primary_or_authoritative = get_first(
        ev,
        ["f4_is_primary_or_authoritative", ["pafc", "f4", "is_primary_or_authoritative"]],
        default=None,
    )
    f4_satire_or_humor = get_first(
        ev,
        ["f4_is_satire_or_humor", ["pafc", "f4", "is_satire_or_humor"]],
        default=None,
    )
    f4_low_credibility_or_abuse = get_first(
        ev,
        ["f4_is_low_credibility_or_abuse", ["pafc", "f4", "is_low_credibility_or_abuse"]],
        default=None,
    )

    return f"""Evidence {idx}
Question: {truncate_text(question, 400)}
Answer: {truncate_text(answer, max_answer_chars)}

f1_relation:
- label: {normalize_text(f1_label) or "nei"}
- p_support: {p_support:.3f}
- p_refute: {p_refute:.3f}
- p_neutral: {p_neutral:.3f}

f2_source_quality:
- source_type: {normalize_text(f2_source_type) or "unknown"}
- leakage_risk: {yes_no(f2_leakage_sensitive)}

f3_trust:
- trust_score: {f3_trust_score:.3f}
- trust_level: {normalize_text(f3_trust_level) or "unknown"}
- trust_reasons: {compact_list(f3_reasons, max_items=3)}

f4_provenance:
- provenance_level: {normalize_text(f4_provenance_level) or "unknown"}
- provenance_flags: {compact_list(f4_provenance_flags, max_items=10)}
- provenance_reasons: {compact_list(f4_provenance_reasons, max_items=3)}
- primary_or_authoritative: {yes_no(f4_primary_or_authoritative)}
- fact_check_article: {yes_no(f4_fact_check_article)}
- satire_or_humor: {yes_no(f4_satire_or_humor)}
- low_credibility_or_abuse: {yes_no(f4_low_credibility_or_abuse)}

source:
- domain: {normalize_text(domain) or "unknown"}
- url: {truncate_text(url, 300) or "unknown"}
"""


def build_dev_indexes(
    dev_records: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    dev_by_id: dict[int, dict[str, Any]] = {}
    dev_by_claim: dict[str, dict[str, Any]] = {}

    for idx, rec in enumerate(dev_records):
        dev_by_id[idx] = rec

        claim_key = normalize_key(rec.get("claim"))
        if claim_key:
            dev_by_claim[claim_key] = rec

    return dev_by_id, dev_by_claim


def find_gold_record(
    retrieval_record: dict[str, Any],
    dev_by_id: dict[int, dict[str, Any]],
    dev_by_claim: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    claim_id = retrieval_record.get("claim_id")
    retrieval_claim_key = normalize_key(retrieval_record.get("claim"))

    if isinstance(claim_id, int) and claim_id in dev_by_id:
        candidate = dev_by_id[claim_id]
        if normalize_key(candidate.get("claim")) == retrieval_claim_key:
            return candidate

    return dev_by_claim.get(retrieval_claim_key)


def build_user_prompt(
    retrieval_record: dict[str, Any],
    dev_record: dict[str, Any],
    top_k: int,
    max_answer_chars: int,
) -> str:
    claim = get_first(retrieval_record, ["claim"], default="")
    if not claim:
        claim = get_first(dev_record, ["claim"], default="")

    claim_date = get_first(dev_record, ["claim_date"], default="unknown")
    speaker = get_first(dev_record, ["speaker"], default="unknown")
    reporting_source = get_first(dev_record, ["reporting_source"], default="unknown")
    location = get_first(dev_record, ["location_ISO_code"], default="unknown")
    claim_types = get_first(dev_record, ["claim_types"], default=[])

    evidence_items = retrieval_record.get("top_10", [])
    if not isinstance(evidence_items, list):
        evidence_items = []

    evidence_texts = []
    for i, ev in enumerate(evidence_items[:top_k], start=1):
        if isinstance(ev, dict):
            evidence_texts.append(build_evidence_item(ev, i, max_answer_chars))

    evidence_block = "\n".join(evidence_texts) if evidence_texts else "No retrieved evidence items available."

    return f"""Claim:
{normalize_text(claim)}

Claim Metadata:
- claim_date: {normalize_text(claim_date) or "unknown"}
- speaker: {normalize_text(speaker) or "unknown"}
- reporting_source: {normalize_text(reporting_source) or "unknown"}
- location_ISO_code: {normalize_text(location) or "unknown"}
- claim_types: {compact_list(claim_types, max_items=8)}

Retrieved Evidence Items:
{evidence_block}

Task:
Use the claim, evidence answer text, and all f1/f2/f3/f4 metadata to output exactly one verdict line.
"""


def make_sft_example(
    retrieval_record: dict[str, Any],
    dev_record: dict[str, Any],
    top_k: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    gold_label = normalize_label(dev_record.get("label"))

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    retrieval_record=retrieval_record,
                    dev_record=dev_record,
                    top_k=top_k,
                    max_answer_chars=max_answer_chars,
                ),
            },
            {"role": "assistant", "content": f"VERDICT: {gold_label}"},
        ]
    }


def split_records(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_ratio <= 0:
        return records, []
    if val_ratio >= 1:
        raise ValueError("--val-ratio must be less than 1.0")

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_size:], shuffled[:val_size]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_dataset_info(dataset_name: str, file_name: str) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert fact-check retrieval data + dev labels into LlamaFactory SFT JSONL."
    )

    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-name", default="factcheck_verdict_sft")
    parser.add_argument("--dataset-info-out", type=Path, default=None)

    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--val-out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-answer-chars", type=int, default=1200)

    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--missing-out", type=Path, default=Path("missing_label_matches.json"))

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    if args.val_ratio > 0 and args.val_out is None:
        raise ValueError("Please provide --val-out when using --val-ratio > 0")

    retrieval_records = read_json_or_jsonl(args.retrieval)
    dev_records = read_json_or_jsonl(args.dev)

    dev_by_id, dev_by_claim = build_dev_indexes(dev_records)

    examples: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()

    for rec_idx, retrieval_record in enumerate(retrieval_records):
        dev_record = find_gold_record(retrieval_record, dev_by_id, dev_by_claim)

        if dev_record is None:
            missing.append(
                {
                    "record_index": rec_idx,
                    "claim_id": retrieval_record.get("claim_id"),
                    "claim": retrieval_record.get("claim"),
                }
            )
            if args.skip_missing:
                continue

            raise ValueError(
                "Could not find matching dev.json label for "
                f"record_index={rec_idx}, "
                f"claim_id={retrieval_record.get('claim_id')}, "
                f"claim={retrieval_record.get('claim')!r}"
            )

        gold_label = normalize_label(dev_record.get("label"))
        label_counter[gold_label] += 1

        examples.append(
            make_sft_example(
                retrieval_record=retrieval_record,
                dev_record=dev_record,
                top_k=args.top_k,
                max_answer_chars=args.max_answer_chars,
            )
        )

    train_examples, val_examples = split_records(examples, args.val_ratio, args.seed)

    write_jsonl(train_examples, args.out)

    if args.val_ratio > 0:
        write_jsonl(val_examples, args.val_out)

    if missing:
        write_json(missing, args.missing_out)

    dataset_info_out = args.dataset_info_out
    if dataset_info_out is None:
        dataset_info_out = args.out.with_suffix(".dataset_info_patch.json")

    dataset_info = build_dataset_info(
        dataset_name=args.dataset_name,
        file_name=args.out.name,
    )
    write_json(dataset_info, dataset_info_out)

    print("Done.")
    print(f"Retrieval records loaded: {len(retrieval_records)}")
    print(f"dev.json records loaded:  {len(dev_records)}")
    print(f"SFT examples created:     {len(examples)}")
    print(f"Train output:             {args.out}")
    print(f"Train examples:           {len(train_examples)}")

    if args.val_ratio > 0:
        print(f"Validation output:        {args.val_out}")
        print(f"Validation examples:      {len(val_examples)}")

    print(f"Dataset info patch:       {dataset_info_out}")

    print("Label distribution:")
    for label, count in sorted(label_counter.items()):
        print(f"  {label}: {count}")

    if missing:
        print(f"Missing matches:          {len(missing)}")
        print(f"Missing written to:       {args.missing_out}")


if __name__ == "__main__":
    main()
