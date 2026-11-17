#!/usr/bin/env python3
"""
Convert verbose PA-FC formatted files (train/dev_format_topK_labeled.jsonl)
into the compact format.

Works directly on the *formatted* files (the ones with `input_text`),
so the examples, labels, and evidence order are guaranteed identical
to the v1 runs -> clean controlled comparison.

What it removes / compresses per evidence item:
  - CLAIM_ID header                          (dropped: spurious id signal)
  - F1 probability triple + 6-dec floats     (kept: relation + conf @ 2dp)
  - F2 match_confidence                      (dropped: pipeline bookkeeping)
  - F3 trust_score float                     (kept: trust_level only)
  - F4 provenance_labels + provenance_score  (dropped: redundant w/ flags)
  - all `is_*: false` boolean lines          (only TRUE flags survive)
  - `has_url` label                          (always true -> no information)
  - QA answer markdown + paraphrase padding  (first N sentences kept)

Output line format per evidence:

  [E1] Q: <question>
  A: <answer, <=N sentences, markdown stripped>
  NLI: nei (0.99) | SRC: est_news | TRUST: medium_high | PROV: moderate | FLAGS: -

Usage:
  python convert_to_compact_format.py \
      --input train_format_top3_labeled.jsonl \
      --output train_format_top3_labeled_compact.jsonl
"""

import argparse
import json
import re
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing the verbose input_text
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^(CLAIM_ID|CLAIM|EVIDENCE_QA_(\d+)|F1_NLI_(\d+)|F2_SOURCE_(\d+)|"
    r"F3_TRUST_(\d+)|F4_PROVENANCE_(\d+)):",
    re.MULTILINE,
)

# F4 boolean field -> compact flag name
FLAG_MAP = {
    "is_archive_or_copy": "archive_copy",
    "is_fact_check_article": "fact_check",
    "is_primary_or_authoritative": "primary_source",
    "is_low_credibility_or_abuse": "low_credibility",
    "is_satire_or_humor": "satire",
    "is_reference_or_data": "reference_data",
    "is_multimedia": "multimedia",
}


def split_sections(input_text: str) -> dict:
    """Splits the verbose input_text into {section_name: body_text}."""
    matches = list(HEADER_RE.finditer(input_text))
    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(input_text)
        sections[name] = input_text[start:end].strip()
    return sections


def parse_kv_block(body: str) -> dict:
    """Parses simple `key: value` lines inside a feature block."""
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip()
    return out


def parse_qa_block(body: str) -> tuple:
    """Extracts question and (multi-line) answer from an EVIDENCE_QA block."""
    q_match = re.search(r"^question:\s*(.*?)$", body, re.MULTILINE)
    question = q_match.group(1).strip() if q_match else ""

    a_match = re.search(r"^answer:\s*(.*)", body, re.MULTILINE | re.DOTALL)
    answer = a_match.group(1).strip() if a_match else ""
    return question, answer


# ---------------------------------------------------------------------------
# Answer compression
# ---------------------------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
ABBREV_FIX = ("U.S.", "U.K.", "Lt.", "Col.", "Dr.", "Mr.", "Mrs.", "Ms.",
              "St.", "Jr.", "Sr.", "Gen.", "Sen.", "Rep.", "Gov.", "vs.",
              "Inc.", "Ltd.", "Co.", "No.", "approx.", "e.g.", "i.e.")


def strip_markdown(text: str) -> str:
    """Removes markdown decoration and bullet structure from LLM answers."""
    # Cut everything after a "context/supporting" header -- it's paraphrase padding.
    cut = re.split(r"\n\s*\*\*[^*]*\*\*\s*:?", text, maxsplit=1)
    text = cut[0] if cut[0].strip() else text

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)      # bold
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)  # bullets
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered
    text = re.sub(r"\s*\n\s*", " ", text)             # newlines -> spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def first_sentences(text: str, n: int) -> str:
    """Keeps the first n sentences, with light abbreviation protection."""
    protected = text
    for abbr in ABBREV_FIX:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    sentences = SENT_SPLIT_RE.split(protected)
    kept = " ".join(sentences[:n]).replace("<DOT>", ".")
    return kept.strip()


def compress_answer(answer: str, max_sentences: int) -> str:
    if not answer:
        return "(no answer)"
    return first_sentences(strip_markdown(answer), max_sentences)


# ---------------------------------------------------------------------------
# Compact rendering
# ---------------------------------------------------------------------------

def round2(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def build_flags(f2: dict, f4: dict) -> str:
    flags = []
    if str(f2.get("leakage_sensitive", "")).lower() == "true":
        flags.append("leakage_sensitive")
    for field, flag in FLAG_MAP.items():
        if str(f4.get(field, "")).lower() == "true":
            flags.append(flag)
    return ", ".join(flags) if flags else "-"


def render_evidence(idx: int, sections: dict, max_sentences: int) -> str:
    qa_body = sections.get(f"EVIDENCE_QA_{idx}", "")
    f1 = parse_kv_block(sections.get(f"F1_NLI_{idx}", ""))
    f2 = parse_kv_block(sections.get(f"F2_SOURCE_{idx}", ""))
    f3 = parse_kv_block(sections.get(f"F3_TRUST_{idx}", ""))
    f4 = parse_kv_block(sections.get(f"F4_PROVENANCE_{idx}", ""))

    question, answer = parse_qa_block(qa_body)
    answer = compress_answer(answer, max_sentences)

    nli = f"{f1.get('relation', 'nei')} ({round2(f1.get('confidence', '0'))})"
    src = f2.get("source_type", "unknown")
    trust = f3.get("trust_level", "unknown")
    prov = f4.get("provenance_level", "unknown")
    flags = build_flags(f2, f4)

    lines = [
        f"[E{idx}] Q: {question if question else '(no question)'}",
        f"A: {answer}",
        f"NLI: {nli} | SRC: {src} | TRUST: {trust} | PROV: {prov} | FLAGS: {flags}",
    ]
    return "\n".join(lines)


def convert_input_text(input_text: str, max_sentences: int) -> str:
    sections = split_sections(input_text)

    claim = sections.get("CLAIM", "").strip()
    if not claim:
        raise ValueError("Could not parse CLAIM section.")

    # Find all evidence indices present (handles any top_k).
    indices = sorted(
        int(m.group(2))
        for m in HEADER_RE.finditer(input_text)
        if m.group(1).startswith("EVIDENCE_QA_")
    )

    parts = [f"CLAIM: {claim}"]
    for idx in indices:
        parts.append(render_evidence(idx, sections, max_sentences))

    return "\n\n".join(parts)


def regex_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Verbose formatted file, e.g. train_format_top3_labeled.jsonl")
    parser.add_argument("--output", required=True,
                        help="Output compact JSONL path.")
    parser.add_argument("--answer-sentences", type=int, default=2,
                        help="Max sentences kept from each QA answer (default 2).")
    parser.add_argument("--show-examples", type=int, default=1,
                        help="Print the first N converted examples for inspection.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    old_counts, new_counts = [], []
    n_records = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            old_text = rec["input_text"]
            try:
                new_text = convert_input_text(old_text, args.answer_sentences)
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: {exc}") from exc

            old_counts.append(regex_token_count(old_text))
            new_counts.append(regex_token_count(new_text))

            out = {
                "claim_id": rec.get("claim_id"),
                "top_k": rec.get("top_k"),
                "input_text": new_text,
                "label": rec.get("label"),
                "label_match_status": rec.get("label_match_status"),
                "token_counts": {"regex_token_count": new_counts[-1]},
                "format_version": "compact_v1",
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            if n_records < args.show_examples:
                print("=" * 70)
                print(f"EXAMPLE {n_records + 1} (claim_id={rec.get('claim_id')}, "
                      f"label={rec.get('label')})")
                print("=" * 70)
                print(new_text)
                print()
            n_records += 1

    print("-" * 70)
    print(f"Converted {n_records} records -> {output_path}")
    print(f"Mean regex tokens: {statistics.mean(old_counts):.0f} -> "
          f"{statistics.mean(new_counts):.0f} "
          f"({100 * (1 - statistics.mean(new_counts) / statistics.mean(old_counts)):.0f}% reduction)")
    print(f"Max regex tokens:  {max(old_counts)} -> {max(new_counts)}")


if __name__ == "__main__":
    main()
