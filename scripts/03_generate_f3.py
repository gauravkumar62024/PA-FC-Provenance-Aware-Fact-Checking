import os as _os, sys as _sys
_PAFC_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _PAFC_SRC not in _sys.path:
    _sys.path.insert(0, _PAFC_SRC)

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except Exception:
    yaml = None


TOP_KEYS = ("top_10", "top_20", "top_30", "top_k", "evidence")


# F3 trust priors by F2 source type.
# These are source-prior scores, not truth labels.
F2_TRUST_PRIORS = {
    "gov": 0.95,
    "government_official": 0.95,

    "academic": 0.90,
    "academic_scientific": 0.90,

    "fact_checker": 0.88,

    "est_news": 0.76,
    "established_news": 0.76,

    "aggregator": 0.55,

    "blog": 0.48,
    "blog_cms": 0.48,

    "forum": 0.40,
    "forum_community": 0.40,

    "social": 0.38,
    "social_platform": 0.38,

    "low_credibility_news": 0.20,
    "cyber_abuse_or_fraud_signal": 0.10,

    "unknown": 0.50,
}


def read_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], "jsonl"

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data, "json_array"

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at line {line_no} in {path}")
        records.append(obj)

    return records, "jsonl"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def find_evidence_key(record: dict[str, Any]) -> str | None:
    for key in TOP_KEYS:
        if isinstance(record.get(key), list):
            return key
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def normalize_domain(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return ""

    if "://" in value:
        parsed = urlparse(value)
        host = parsed.netloc.lower()
    else:
        host = value.split("/")[0].lower()

    host = host.split("@")[-1].split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]

    if not host:
        return ""

    try:
        import tldextract

        ext = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=False)(host)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        pass

    parts = host.split(".")
    if len(parts) <= 2:
        return host

    two_part_suffixes = {
        "co.uk",
        "ac.uk",
        "gov.uk",
        "com.au",
        "gov.au",
        "edu.au",
        "co.in",
        "gov.in",
        "com.ph",
        "com.ng",
    }
    suffix2 = ".".join(parts[-2:])
    if suffix2 in two_part_suffixes and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def get_row_domain(ev: dict[str, Any]) -> str:
    for key in (
        "domain",
        "registered_domain",
        "original_domain",
        "canonical_source_url",
        "url",
        "retrieved_url",
    ):
        value = ev.get(key)
        if value:
            domain = normalize_domain(str(value))
            if domain:
                return domain
    return ""


def trust_level(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium_high"
    if score >= 0.45:
        return "medium"
    if score >= 0.25:
        return "low"
    return "very_low"


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def load_domain_overrides(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []

    suffix = path.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed, but a YAML trust file was provided.")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = data.get("entries", data if isinstance(data, list) else [])
        if not isinstance(entries, list):
            return []
        return [dict(item) for item in entries if isinstance(item, dict)]

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries", data if isinstance(data, list) else [])
        if not isinstance(entries, list):
            return []
        return [dict(item) for item in entries if isinstance(item, dict)]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]

    return []


def find_default_trust_file() -> Path | None:
    candidates = [
        Path("configs/source_lists/f3/domain_trust.yml"),
        Path("configs/f3_domain_trust.yml"),
        Path("f3_domain_trust.yml"),
        Path("averitec_f2_expanded_domain_package/f3_domain_trust_review.yml"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def match_override(domain: str, url: str, overrides: list[dict[str, Any]]) -> dict[str, Any] | None:
    domain = (domain or "").lower().strip()
    url = (url or "").lower().strip()

    best = None
    best_score = -1

    for item in overrides:
        pattern = str(item.get("pattern", item.get("domain", ""))).lower().strip()
        if not pattern:
            continue

        match_type = str(item.get("match_type", "registered_domain")).lower().strip()
        matched = False

        if match_type == "registered_domain":
            pat = pattern.strip(".")
            matched = domain == pat or domain.endswith("." + pat)

        elif match_type == "suffix":
            pat = pattern if pattern.startswith(".") else "." + pattern
            bare = pat.lstrip(".")
            matched = domain == bare or domain.endswith(pat)

        elif match_type == "contains":
            matched = pattern in domain or pattern in url

        elif match_type == "url_contains":
            matched = pattern in url

        elif match_type == "fallback":
            matched = True

        if matched:
            priority = {
                "registered_domain": 100,
                "suffix": 80,
                "contains": 50,
                "url_contains": 40,
                "fallback": 1,
            }.get(match_type, 10)
            score = priority + len(pattern)
            if score > best_score:
                best = item
                best_score = score

    return best


def get_f2_source_type(ev: dict[str, Any]) -> str:
    value = ev.get("f2_source_type")
    if value:
        return str(value).strip()

    pafc = ev.get("pafc")
    if isinstance(pafc, dict):
        f2 = pafc.get("f2")
        if isinstance(f2, dict):
            value = f2.get("source_type")
            if value:
                return str(value).strip()

    return "unknown"


def get_f2_match_confidence(ev: dict[str, Any]) -> str:
    value = ev.get("f2_match_confidence")
    if value:
        return str(value).strip().lower()

    pafc = ev.get("pafc")
    if isinstance(pafc, dict):
        f2 = pafc.get("f2")
        if isinstance(f2, dict):
            value = f2.get("match_confidence")
            if value:
                return str(value).strip().lower()

    return "low"


def bucket_default_score(f2_source_type: str, f2_match_confidence: str) -> tuple[float, list[str]]:
    source_type = (f2_source_type or "unknown").strip()
    confidence = (f2_match_confidence or "low").strip().lower()

    base = F2_TRUST_PRIORS.get(source_type, 0.50)
    reasons = [f"bucket_prior:{source_type}={base:.2f}"]

    if confidence == "high":
        base += 0.02
        reasons.append("f2_match_confidence_high:+0.02")
    elif confidence in {"medium", "dataset_audit"}:
        reasons.append(f"f2_match_confidence_{confidence}:no_change")
    elif confidence == "low":
        base -= 0.03
        reasons.append("f2_match_confidence_low:-0.03")

    return clamp_score(base), reasons


def compute_f3_for_evidence(ev: dict[str, Any], overrides: list[dict[str, Any]]) -> dict[str, Any]:
    domain = get_row_domain(ev)
    url = str(ev.get("canonical_source_url") or ev.get("url") or "")
    f2_source_type = get_f2_source_type(ev)
    f2_match_confidence = get_f2_match_confidence(ev)

    override = match_override(domain, url, overrides)

    if override is not None:
        score = clamp_score(safe_float(override.get("trust_score", override.get("score", 0.5)), 0.5))
        source = str(override.get("source", override.get("source_file", "domain_override")))
        notes = str(override.get("notes", override.get("reason", "")))
        level = str(override.get("trust_level", trust_level(score)))

        reasons = [f"domain_override:{domain}={score:.2f}"]
        if notes:
            reasons.append(notes)

        return {
            "domain": domain,
            "trust_score": round(score, 6),
            "trust_level": level,
            "trust_source": source,
            "trust_method": "domain_override",
            "reasons": reasons,
        }

    score, reasons = bucket_default_score(f2_source_type, f2_match_confidence)

    return {
        "domain": domain,
        "trust_score": round(score, 6),
        "trust_level": trust_level(score),
        "trust_source": "f2_bucket_default",
        "trust_method": "f2_source_type_prior",
        "reasons": reasons,
    }


def attach_f3(records: list[dict[str, Any]], overrides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for claim_rec in records:
        evidence_key = find_evidence_key(claim_rec)
        if evidence_key is None:
            continue

        evidences = claim_rec.get(evidence_key, [])
        if not isinstance(evidences, list):
            continue

        for ev in evidences:
            if not isinstance(ev, dict):
                continue

            f3 = compute_f3_for_evidence(ev, overrides)

            ev["f3_domain"] = f3["domain"]
            ev["f3_trust_score"] = f3["trust_score"]
            ev["f3_trust_level"] = f3["trust_level"]
            ev["f3_trust_source"] = f3["trust_source"]
            ev["f3_trust_method"] = f3["trust_method"]
            ev["f3_trust_reasons"] = f3["reasons"]

            pafc = ev.get("pafc")
            if not isinstance(pafc, dict):
                pafc = {}
            pafc["f3"] = {
                "domain": f3["domain"],
                "trust_score": f3["trust_score"],
                "trust_level": f3["trust_level"],
                "trust_source": f3["trust_source"],
                "trust_method": f3["trust_method"],
                "reasons": f3["reasons"],
            }
            ev["pafc"] = pafc

    return records


def write_audit_files(output_dir: Path, input_stem: str, records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for claim_rec in records:
        claim_id = claim_rec.get("claim_id", claim_rec.get("id", ""))
        claim = claim_rec.get("claim", "")
        evidence_key = find_evidence_key(claim_rec)

        if evidence_key is None:
            continue

        for ev in claim_rec.get(evidence_key, []):
            if not isinstance(ev, dict):
                continue

            rows.append({
                "claim_id": claim_id,
                "claim": claim,
                "rank": ev.get("rank"),
                "source_doc_idx": ev.get("source_doc_idx"),
                "retrieval_score": ev.get("retrieval_score"),
                "url": ev.get("url"),
                "canonical_source_url": ev.get("canonical_source_url"),
                "domain": ev.get("domain"),
                "f2_source_type": ev.get("f2_source_type"),
                "f2_match_source": ev.get("f2_match_source"),
                "f2_match_confidence": ev.get("f2_match_confidence"),
                "f3_domain": ev.get("f3_domain"),
                "f3_trust_score": ev.get("f3_trust_score"),
                "f3_trust_level": ev.get("f3_trust_level"),
                "f3_trust_source": ev.get("f3_trust_source"),
                "f3_trust_method": ev.get("f3_trust_method"),
                "f1_relation_label": ev.get("f1_relation_label"),
                "f1_confidence": ev.get("f1_confidence"),
                "sentence": ev.get("sentence", ""),
            })

    audit_path = output_dir / f"{input_stem}_f3_audit.csv"
    summary_path = output_dir / f"{input_stem}_f3_domain_summary.csv"

    if rows:
        fieldnames = list(rows[0].keys())
        with audit_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        domain_counter = Counter()
        domain_score_sum = Counter()

        for row in rows:
            key = (
                row["f3_domain"],
                row["f2_source_type"],
                row["f3_trust_level"],
                row["f3_trust_source"],
            )
            domain_counter[key] += 1
            domain_score_sum[key] += safe_float(row["f3_trust_score"], 0.0)

        summary_rows = []
        for key, count in domain_counter.items():
            domain, f2_source_type, level, source = key
            mean_score = domain_score_sum[key] / count if count else 0.0
            summary_rows.append({
                "domain": domain,
                "f2_source_type": f2_source_type,
                "f3_trust_level": level,
                "f3_trust_source": source,
                "count": count,
                "mean_trust_score": round(mean_score, 6),
            })

        summary_rows.sort(key=lambda x: (-x["count"], x["domain"] or ""))

        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="f2_feature_data/retrieval_top_k_with_idx_f1_f2.json",
        help="Input JSON/JSONL file containing F1 and F2 features.",
    )
    parser.add_argument(
        "--output-dir",
        default="f3_feature_data",
        help="Output folder.",
    )
    parser.add_argument(
        "--output-name",
        default="retrieval_top_k_with_idx_f1_f2_f3.json",
        help="Output JSONL file name.",
    )
    parser.add_argument(
        "--domain-trust-file",
        default=None,
        help="Optional YAML/JSON/CSV file with domain trust overrides.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    trust_file = Path(args.domain_trust_file) if args.domain_trust_file else find_default_trust_file()
    overrides = load_domain_overrides(trust_file) if trust_file else []

    print("F3 trust configuration:")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    print(f"  domain_trust_file: {trust_file if trust_file else 'none'}")
    print(f"  domain_override_rules: {len(overrides)}")

    records, input_format = read_json_or_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    records = attach_f3(records, overrides)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    write_audit_files(output_dir, input_path.stem, records)

    trust_counts = Counter()
    source_counts = Counter()
    row_count = 0

    for claim_rec in records:
        evidence_key = find_evidence_key(claim_rec)
        if evidence_key is None:
            continue
        for ev in claim_rec.get(evidence_key, []):
            if not isinstance(ev, dict):
                continue
            row_count += 1
            trust_counts[str(ev.get("f3_trust_level"))] += 1
            source_counts[str(ev.get("f3_trust_source"))] += 1

    print("\nDone.")
    print(f"Input format detected: {input_format}")
    print(f"Claim rows: {len(records)}")
    print(f"Evidence rows scored: {row_count}")
    print(f"Output: {output_path}")
    print(f"Audit CSV: {output_dir / (input_path.stem + '_f3_audit.csv')}")
    print(f"Domain summary CSV: {output_dir / (input_path.stem + '_f3_domain_summary.csv')}")

    print("\nF3 trust-level counts:")
    for label, count in trust_counts.most_common():
        print(f"  {label}: {count}")

    print("\nF3 trust-source counts:")
    for label, count in source_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
