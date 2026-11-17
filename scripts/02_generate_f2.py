import os as _os, sys as _sys
_PAFC_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _PAFC_SRC not in _sys.path:
    _sys.path.insert(0, _PAFC_SRC)

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote
from tqdm import tqdm
import pandas as pd


TOP_KEYS = ("top_10", "top_20", "top_30", "top_k", "evidence")


def add_repo_to_path() -> None:
    """
    Makes sure Python can import pa_fc when this script is run from the project folder.
    """
    here = Path.cwd().resolve()
    candidates = [
        here,
        here / "pa_fc",
        here.parent,
        here.parent / "pa_fc",
    ]

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def find_source_list_dir(user_dir: str | None = None) -> Path | None:
    """
    Priority:
    1. --source-list-dir
    2. PA_FC_F2_SOURCE_LIST_DIR env var
    3. ./averitec_f2_expanded_domain_package/f2_source_lists_augmented_review
    4. ./configs/source_lists/f2
    5. ./pa_fc/configs/source_lists/f2
    """
    if user_dir:
        path = Path(user_dir).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"--source-list-dir does not exist: {path}")

    env_dir = os.environ.get("PA_FC_F2_SOURCE_LIST_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"PA_FC_F2_SOURCE_LIST_DIR does not exist: {path}")

    here = Path.cwd().resolve()
    candidates = [
        here / "averitec_f2_expanded_domain_package" / "f2_source_lists_augmented_review",
        here / "configs" / "source_lists" / "f2",
        here / "pa_fc" / "configs" / "source_lists" / "f2",
        here.parent / "configs" / "source_lists" / "f2",
        here.parent / "pa_fc" / "configs" / "source_lists" / "f2",
    ]

    for path in candidates:
        if path.exists() and any(path.glob("*.yml")):
            return path.resolve()

    return None


def read_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    """
    Returns records and format type: json_array or jsonl.
    """
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


def unwrap_archive_url(url: str) -> tuple[str, str, bool]:
    """
    Returns:
      canonical_url, archive_url, is_archive_url

    Keeps archive wrapper separately, but classifies the real target URL when possible.
    """
    if not url:
        return "", "", False

    raw = str(url).strip()
    parsed = urlparse(raw)
    host = parsed.netloc.lower().replace("www.", "")

    # web.archive.org/web/<timestamp>/<target_url>
    if "web.archive.org" in host and "/web/" in parsed.path:
        parts = parsed.path.split("/web/", 1)
        if len(parts) == 2:
            rest = parts[1].lstrip("/")
            rest_parts = rest.split("/", 1)
            if len(rest_parts) == 2:
                target = rest_parts[1]
                target = unquote(target)
                if target.startswith(("http://", "https://")):
                    return target, raw, True

    # archive.today / archive.ph / archive.is sometimes store target in query,
    # but if we cannot safely extract it, keep original as canonical.
    archive_hosts = {
        "archive.ph",
        "archive.is",
        "archive.today",
        "archive.vn",
        "perma.cc",
    }
    if host in archive_hosts or any(host.endswith("." + h) for h in archive_hosts):
        return raw, raw, True

    return raw, "", False


def simple_registered_domain(url_or_domain: str) -> str:
    """
    Lightweight registered-domain extraction.
    Uses tldextract if available, otherwise simple fallback.
    """
    value = (url_or_domain or "").strip()
    if not value:
        return ""

    if "://" not in value:
        host = value.lower().split("/")[0]
    else:
        parsed = urlparse(value)
        host = parsed.netloc.lower()

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


def flatten_records(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[tuple[int, str, int]]]:
    rows = []
    index_map = []

    for claim_pos, claim_rec in enumerate(records):
        claim_id = claim_rec.get("claim_id", claim_rec.get("id", claim_pos))
        claim_text = claim_rec.get("claim", "")

        evidence_key = find_evidence_key(claim_rec)
        if evidence_key is None:
            continue

        evidences = claim_rec.get(evidence_key, [])
        for ev_pos, ev in enumerate(evidences):
            if not isinstance(ev, dict):
                continue

            original_url = (
                ev.get("canonical_source_url")
                or ev.get("source_url")
                or ev.get("url")
                or ""
            )

            canonical_url, archive_url, is_archive_url = unwrap_archive_url(str(original_url))
            domain = ev.get("domain") or ev.get("registered_domain") or ""
            canonical_domain = simple_registered_domain(canonical_url) or simple_registered_domain(domain)

            # Add canonical fields immediately so they are also present in output.
            ev.setdefault("retrieved_url", ev.get("url", original_url))
            ev["canonical_source_url"] = canonical_url
            ev["archive_url"] = archive_url
            ev["is_archive_url"] = bool(is_archive_url)
            ev["original_domain"] = simple_registered_domain(ev.get("retrieved_url", original_url))
            ev["domain"] = canonical_domain

            rows.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "evidence_key": evidence_key,
                    "evidence_pos": ev_pos,

                    # Fields expected by compute_f2
                    "source_url": canonical_url,
                    "canonical_source_url": canonical_url,
                    "registered_domain": canonical_domain,
                    "source_medium_raw": ev.get("source_medium_raw", "Web text"),
                    "source_type_raw": ev.get("source_type_raw", "unknown"),

                    # Audit fields
                    "rank": ev.get("rank"),
                    "source_doc_idx": ev.get("source_doc_idx"),
                    "retrieval_score": ev.get("retrieval_score"),
                    "url": ev.get("url", ""),
                    "archive_url": archive_url,
                    "is_archive_url": bool(is_archive_url),
                    "sentence": ev.get("sentence", ""),
                    "f1_relation_label": ev.get("f1_relation_label", ""),
                    "f1_confidence": ev.get("f1_confidence", ""),
                }
            )

            index_map.append((claim_pos, evidence_key, ev_pos))

    return pd.DataFrame(rows), index_map


def attach_f2_to_records(
    records: list[dict[str, Any]],
    f2_df: pd.DataFrame,
    index_map: list[tuple[int, str, int]],
) -> None:
    for row_idx, (claim_pos, evidence_key, ev_pos) in tqdm(enumerate(index_map), total=len(index_map), desc="Attaching F2 features"):
        ev = records[claim_pos][evidence_key][ev_pos]
        row = f2_df.iloc[row_idx]

        f2_block = {
            "source_type": str(row["f2_source_type"]),
            "match_source": str(row["f2_match_source"]),
            "match_confidence": str(row["f2_match_confidence"]),
            "leakage_sensitive": bool(row["f2_leakage_sensitive"]),
        }

        # Flat fields
        ev["f2_source_type"] = f2_block["source_type"]
        ev["f2_match_source"] = f2_block["match_source"]
        ev["f2_match_confidence"] = f2_block["match_confidence"]
        ev["f2_leakage_sensitive"] = f2_block["leakage_sensitive"]

        # Nested PA-FC block
        pafc = ev.get("pafc")
        if not isinstance(pafc, dict):
            pafc = {}

        pafc["f2"] = f2_block
        ev["pafc"] = pafc


def write_audit_files(output_dir: Path, input_stem: str, f2_df: pd.DataFrame, runtime: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_cols = [
        "claim_id",
        "rank",
        "source_doc_idx",
        "retrieval_score",
        "url",
        "canonical_source_url",
        "archive_url",
        "is_archive_url",
        "registered_domain",
        "f2_source_type",
        "f2_match_source",
        "f2_match_confidence",
        "f2_leakage_sensitive",
        "f1_relation_label",
        "f1_confidence",
        "sentence",
    ]
    audit_cols = [c for c in audit_cols if c in f2_df.columns]
    f2_df[audit_cols].to_csv(output_dir / f"{input_stem}_f2_audit.csv", index=False)

    domain_summary = (
        f2_df.groupby(["registered_domain", "f2_source_type", "f2_match_source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["f2_source_type", "count"], ascending=[True, False])
    )
    domain_summary.to_csv(output_dir / f"{input_stem}_f2_domain_summary.csv", index=False)

    unknowns = (
        f2_df[f2_df["f2_source_type"].astype(str) == "unknown"]
        .groupby(["registered_domain"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    unknowns.to_csv(output_dir / f"{input_stem}_f2_unknown_domains.csv", index=False)

    (output_dir / "f2_runtime_info.json").write_text(
        json.dumps(runtime, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="f1_feature_data/retrieval_top_k_with_idx_f1.json",
        help="Input JSON/JSONL file with F1 features already added.",
    )
    parser.add_argument(
        "--output-dir",
        default="f2_feature_data",
        help="Output folder.",
    )
    parser.add_argument(
        "--output-name",
        default="retrieval_top_k_with_idx_f1_f2.json",
        help="Output JSONL file name.",
    )
    parser.add_argument(
        "--source-list-dir",
        default=None,
        help="Optional path to F2 YAML source-list directory.",
    )
    args = parser.parse_args()

    add_repo_to_path()

    source_list_dir = find_source_list_dir(args.source_list_dir)
    if source_list_dir is not None:
        os.environ["PA_FC_F2_SOURCE_LIST_DIR"] = str(source_list_dir)

    from pa_fc.features.f2_source_type import compute_f2, source_list_runtime_info

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    runtime = source_list_runtime_info()

    print("F2 source-list runtime info:")
    print(json.dumps(runtime, indent=2))

    records, input_format = read_json_or_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    flat_df, index_map = flatten_records(records)
    if flat_df.empty:
        raise ValueError("No evidence rows found. Expected top_10/top_20/top_30/top_k/evidence.")

    f2_df = compute_f2(flat_df)
    attach_f2_to_records(records, f2_df, index_map)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    write_audit_files(output_dir, input_path.stem, f2_df, runtime)

    counts = Counter(f2_df["f2_source_type"].astype(str).tolist())
    unknown_count = counts.get("unknown", 0)

    print("\nDone.")
    print(f"Input: {input_path}")
    print(f"Input format detected: {input_format}")
    print(f"Claim rows: {len(records)}")
    print(f"Evidence rows scored: {len(f2_df)}")
    print(f"Output: {output_path}")
    print(f"Unknown rows: {unknown_count} / {len(f2_df)}")

    print("\nF2 source type counts:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
