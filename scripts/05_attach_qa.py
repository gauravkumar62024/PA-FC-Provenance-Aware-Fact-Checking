#!/usr/bin/env python3
import os as _os, sys as _sys
_PAFC_SRC = _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))), "src")
if _PAFC_SRC not in _sys.path:
    _sys.path.insert(0, _PAFC_SRC)

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, urlunparse, unquote

def read_json_or_jsonl(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def unwrap_archive_url(url):
    if not url:
        return ""
    raw = str(url).strip()
    parsed = urlparse(raw)
    host = parsed.netloc.lower().replace("www.", "")
    if "web.archive.org" in host and "/web/" in parsed.path:
        rest = parsed.path.split("/web/", 1)[1].lstrip("/")
        parts = rest.split("/", 1)
        if len(parts) == 2:
            target = unquote(parts[1])
            if target.startswith(("http://", "https://")):
                return target
    return raw

def normalize_url(url, keep_query=True, keep_scheme=True):
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = unquote(parsed.path or "")
        if path != "/":
            path = path.rstrip("/")
        query = parsed.query if keep_query else ""
        if keep_scheme:
            out = urlunparse((scheme, netloc, path, "", query, ""))
        else:
            out = urlunparse(("", netloc, path, "", query, ""))
        return out.lower().lstrip("//").strip()
    except Exception:
        return str(url).strip().lower().rstrip("/")

def url_variants(url):
    if not url:
        return []
    candidates = [str(url).strip()]
    unwrapped = unwrap_archive_url(url)
    if unwrapped and unwrapped != url:
        candidates.append(unwrapped)
    variants = []
    for c in candidates:
        variants.append(normalize_url(c, True, True))
        variants.append(normalize_url(c, False, True))
        variants.append(normalize_url(c, True, False))
        variants.append(normalize_url(c, False, False))
    seen, final = set(), []
    for v in variants:
        if v and v not in seen:
            final.append(v)
            seen.add(v)
    return final

def evidence_url_variants(ev):
    out = []
    for k in ["url", "retrieved_url", "canonical_source_url", "archive_url"]:
        out.extend(url_variants(ev.get(k)))
    seen, final = set(), []
    for v in out:
        if v and v not in seen:
            final.append(v)
            seen.add(v)
    return final

def find_evidence_key(rec, requested=None):
    if requested and isinstance(rec.get(requested), list):
        return requested
    for k in ["top_10", "top_20", "top_30", "top_k", "evidence"]:
        if isinstance(rec.get(k), list):
            return k
    return None

def build_qa_lookup(qa_records):
    url_lookup = {}
    pos_lookup = {}
    for rec in qa_records:
        cid = rec.get("claim_id")
        evs = rec.get("evidence", [])
        for pos, qa in enumerate(evs):
            info = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "url": qa.get("url", ""),
                "qa_position": pos,
            }
            pos_lookup[(cid, pos)] = info
            for v in url_variants(info["url"]):
                url_lookup[(cid, v)] = info
    return url_lookup, pos_lookup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-input", required=True)
    ap.add_argument("--qa-input", required=True)
    ap.add_argument("--output-dir", default="f4_qa_feature_data")
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--evidence-key", default=None)
    args = ap.parse_args()

    feats = read_json_or_jsonl(args.features_input)
    qas = read_json_or_jsonl(args.qa_input)
    url_lookup, pos_lookup = build_qa_lookup(qas)

    counts = Counter()
    audit = []

    for rec in feats:
        cid = rec.get("claim_id")
        key = find_evidence_key(rec, args.evidence_key)
        if key is None:
            counts["missing_evidence_key"] += 1
            continue

        for pos, ev in enumerate(rec.get(key, [])):
            info = None
            status = "unmatched"
            matched = ""

            for v in evidence_url_variants(ev):
                info = url_lookup.get((cid, v))
                if info is not None:
                    status = "matched_by_url"
                    matched = v
                    break

            if info is None:
                info = pos_lookup.get((cid, pos))
                if info is not None:
                    status = "matched_by_position_fallback"
                    matched = f"position:{pos}"

            if info is None:
                ev["question"] = ""
                ev["answer"] = ""
                ev["qa_url"] = ""
                counts["unmatched"] += 1
            else:
                ev["question"] = info["question"]
                ev["answer"] = info["answer"]
                ev["qa_url"] = info["url"]
                counts["matched"] += 1
                counts[status] += 1

            ev["qa_match_status"] = status
            ev["qa_matched_key"] = matched
            if not isinstance(ev.get("pafc"), dict):
                ev["pafc"] = {}
            ev["pafc"]["qa"] = {
                "question": ev.get("question", ""),
                "answer": ev.get("answer", ""),
                "url": ev.get("qa_url", ""),
                "match_status": ev.get("qa_match_status", ""),
                "matched_key": ev.get("qa_matched_key", ""),
            }

            audit.append({
                "claim_id": cid,
                "position": pos,
                "rank": ev.get("rank"),
                "source_doc_idx": ev.get("source_doc_idx"),
                "feature_url": ev.get("url"),
                "qa_url": ev.get("qa_url"),
                "qa_match_status": ev.get("qa_match_status"),
                "has_question": bool(ev.get("question")),
                "has_answer": bool(ev.get("answer")),
            })

    out_dir = Path(args.output_dir)
    out_path = out_dir / args.output_name
    write_jsonl(out_path, feats)

    audit_path = out_dir / f"{out_path.stem}_qa_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        if audit:
            w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
            w.writeheader()
            w.writerows(audit)

    summary_path = out_dir / f"{out_path.stem}_qa_summary.json"
    summary_path.write_text(json.dumps(dict(counts), indent=2), encoding="utf-8")

    print("Output:", out_path)
    print("Counts:", dict(counts))

if __name__ == "__main__":
    main()
