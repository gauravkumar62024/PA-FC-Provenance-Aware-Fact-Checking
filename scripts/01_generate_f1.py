#!/usr/bin/env python3
"""Add F1 claim-evidence relation features to HerO2 retrieval output."""

import os as _os, sys as _sys
_PAFC_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _PAFC_SRC not in _sys.path:
    _sys.path.insert(0, _PAFC_SRC)

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
MODEL_TYPE = "cross_encoder_nli"
PAIR_DIRECTION = "evidence_claim"


def softmax(scores: Any) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    exp = np.exp(arr - np.max(arr))
    denom = exp.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return exp / denom


def read_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], "jsonl"
    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data, "json"
            if isinstance(data, dict):
                return [data], "json"
        except json.JSONDecodeError:
            pass
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            records.append(item)
    return records, "jsonl"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_relation_label(label: str) -> str:
    label = str(label).lower().strip()
    if "entail" in label or "support" in label:
        return "support"
    if "contrad" in label or "refute" in label:
        return "refute"
    if "neutral" in label or "nei" in label:
        return "nei"
    if label in {"label_0", "0"}:
        return "refute"
    if label in {"label_1", "1"}:
        return "support"
    if label in {"label_2", "2"}:
        return "nei"
    return label


def infer_label_order(cross_encoder: Any) -> list[str]:
    fallback = ["refute", "support", "nei"]
    try:
        id2label = getattr(cross_encoder.model.config, "id2label", None)
        if not id2label:
            return fallback
        order = [normalize_relation_label(id2label[idx]) for idx in range(len(id2label))]
        if set(order) == {"support", "refute", "nei"}:
            return order
        print(f"[WARN] Unusual model label order {order}; using fallback {fallback}.")
        return fallback
    except Exception as exc:
        print(f"[WARN] Could not infer model label order: {exc}. Using fallback {fallback}.")
        return fallback


def scores_to_probs(scores: Any, label_order: list[str]) -> tuple[float, float, float]:
    probs = softmax(scores)
    mapped = {"support": 0.0, "refute": 0.0, "nei": 0.0}
    for idx, label in enumerate(label_order):
        if idx < len(probs) and label in mapped:
            mapped[label] = float(probs[idx])
    total = mapped["support"] + mapped["refute"] + mapped["nei"]
    if total <= 0 or not math.isfinite(total):
        return 0.0, 0.0, 1.0
    return mapped["support"] / total, mapped["refute"] / total, mapped["nei"] / total


def label_from_probs(p_support: float, p_refute: float, p_neutral: float) -> str:
    values = {"support": float(p_support), "refute": float(p_refute), "nei": float(p_neutral)}
    return max(values, key=values.get)


def neutral_f1_features(reason: str, model_name: str) -> dict[str, Any]:
    return {
        "f1_p_support": 0.0,
        "f1_p_refute": 0.0,
        "f1_p_neutral": 1.0,
        "f1_relation_label": "nei",
        "f1_confidence": 0.0,
        "f1_status": "skipped",
        "f1_skip_reason": reason,
        "f1_used_nli": False,
        "f1_model_name": model_name,
        "f1_model_type": MODEL_TYPE,
        "f1_pair_direction": PAIR_DIRECTION,
    }


def scored_f1_features(p_support: float, p_refute: float, p_neutral: float, model_name: str) -> dict[str, Any]:
    label = label_from_probs(p_support, p_refute, p_neutral)
    confidence = max(float(p_support), float(p_refute), float(p_neutral))
    return {
        "f1_p_support": round(float(p_support), 6),
        "f1_p_refute": round(float(p_refute), 6),
        "f1_p_neutral": round(float(p_neutral), 6),
        "f1_relation_label": label,
        "f1_confidence": round(confidence, 6),
        "f1_status": "scored",
        "f1_skip_reason": "",
        "f1_used_nli": True,
        "f1_model_name": model_name,
        "f1_model_type": MODEL_TYPE,
        "f1_pair_direction": PAIR_DIRECTION,
    }


def get_claim_text(record: dict[str, Any]) -> str:
    return str(record.get("claim") or record.get("claim_text") or record.get("claim_statement") or "").strip()


def get_evidence_text(evidence: dict[str, Any]) -> str:
    question = str(evidence.get("question") or evidence.get("question_text") or "").strip()
    answer = str(evidence.get("answer") or evidence.get("answer_text") or "").strip()
    if question and answer:
        return f"Question: {question}\nAnswer: {answer}".strip()
    return str(
        evidence.get("sentence")
        or evidence.get("url2text")
        or evidence.get("summary")
        or evidence.get("document_summary")
        or ""
    ).strip()


def skip_reason(claim_text: str, evidence_text: str) -> str | None:
    if not claim_text:
        return "missing_claim"
    norm = evidence_text.lower().strip(" .,:;!?()[]{}\"'")
    compact = "".join(norm.split())
    unanswerable = {
        "",
        "n/a",
        "na",
        "none",
        "null",
        "no answer",
        "no answer could be found",
        "no evidence",
        "not available",
        "unavailable",
        "unanswerable",
        "cannot answer",
        "cannot determine",
        "not enough information",
        "insufficient evidence",
        "unknown",
        "dead link",
        "404",
        "error",
    }
    if norm in unanswerable or len(compact) < 3:
        return "empty_or_unanswerable"
    return None


def detect_evidence_key(record: dict[str, Any], preferred_key: str | None = None) -> str:
    if preferred_key and preferred_key in record and isinstance(record[preferred_key], list):
        return preferred_key
    for key in ("top_10", "top_20", "top_30", "evidence"):
        if key in record and isinstance(record[key], list):
            return key
    for key, value in record.items():
        if key.startswith("top_") and isinstance(value, list):
            return key
    raise KeyError("Could not find an evidence list key. Expected top_10/top_20/top_30/evidence.")


def make_pair(evidence_text: str, claim_text: str) -> tuple[str, str]:
    return evidence_text, claim_text


def add_f1_features(records: list[dict[str, Any]], model_name: str, batch_size: int, evidence_key_arg: str | None) -> dict[str, Any]:
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError("Missing dependency. Install with: pip install sentence-transformers torch") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading F1 model: {model_name} on {device}")
    cross_encoder = CrossEncoder(model_name, max_length=512, device=device)
    label_order = infer_label_order(cross_encoder)
    print(f"[INFO] Model label order mapped to: {label_order}")

    pairs: list[tuple[str, str]] = []
    targets: list[tuple[int, str, int]] = []
    stats = Counter()

    for rec_idx, record in enumerate(records):
        claim_text = get_claim_text(record)
        evidence_key = detect_evidence_key(record, evidence_key_arg)
        top_items = record.get(evidence_key) or []
        stats["claim_rows"] += 1
        stats["evidence_rows"] += len(top_items)
        for ev_idx, evidence in enumerate(top_items):
            if not isinstance(evidence, dict):
                continue
            evidence_text = get_evidence_text(evidence)
            reason = skip_reason(claim_text, evidence_text)
            if reason:
                evidence.update(neutral_f1_features(reason, model_name))
                stats[f"skipped_{reason}"] += 1
                stats["skipped"] += 1
                continue
            pairs.append(make_pair(evidence_text, claim_text))
            targets.append((rec_idx, evidence_key, ev_idx))

    stats["to_score"] = len(pairs)

    if pairs:
        print(f"[INFO] Scoring {len(pairs)} evidence rows in batches of {batch_size}...")
        raw_scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=True)
        for (rec_idx, evidence_key, ev_idx), scores in zip(targets, raw_scores):
            p_support, p_refute, p_neutral = scores_to_probs(scores, label_order)
            features = scored_f1_features(p_support, p_refute, p_neutral, model_name)
            records[rec_idx][evidence_key][ev_idx].update(features)
            stats["scored"] += 1
            stats[f"label_{features['f1_relation_label']}"] += 1

    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add F1 relation features to retrieval_top_k_with_idx.json.")
    parser.add_argument("--input", default="retrieval_top_k_with_idx.json")
    parser.add_argument("--output-dir", default="f1_feature_data")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evidence-key", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_name = args.output_name or f"{input_path.stem}_f1.json"
    output_path = output_dir / output_name
    audit_path = output_dir / f"{input_path.stem}_f1_audit.json"

    print(f"[INFO] Reading: {input_path}")
    records, input_format = read_json_or_jsonl(input_path)
    print(f"[INFO] Loaded {len(records)} claim rows. Detected input format: {input_format}")

    stats = add_f1_features(records, args.model_name, args.batch_size, args.evidence_key)

    print(f"[INFO] Writing F1-feature data to: {output_path}")
    write_jsonl(output_path, records)

    audit = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_format_detected": input_format,
        "model_name": args.model_name,
        "model_type": MODEL_TYPE,
        "pair_direction": PAIR_DIRECTION,
        "stats": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Wrote audit file: {audit_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
