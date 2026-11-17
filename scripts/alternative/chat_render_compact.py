#!/usr/bin/env python3
"""
Convert PA-FC top-k evidence JSONL files into a compact format.

Supports BOTH schemas:
  1) records with an `input_text` field containing headers like CLAIM, EVIDENCE_QA_1, F1_NLI_1...
  2) chat/fine-tuning records with `messages`, where the user message contains:
       Claim:, Claim Metadata:, Retrieved Evidence Items:, Evidence 1..., Task:

For chat/fine-tuning records, this keeps system/assistant messages unchanged and only compacts
retrieved evidence in the user message.
"""

import argparse
import json
import re
import statistics
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Generic text compression helpers
# ---------------------------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
ABBREV_FIX = (
    "U.S.", "U.K.", "Lt.", "Col.", "Dr.", "Mr.", "Mrs.", "Ms.",
    "St.", "Jr.", "Sr.", "Gen.", "Sen.", "Rep.", "Gov.", "vs.",
    "Inc.", "Ltd.", "Co.", "No.", "approx.", "e.g.", "i.e.", "Fig.",
)


def strip_markdown(text: str) -> str:
    """Remove common markdown/padding from evidence answers."""
    if not text:
        return ""

    # Remove common LLM padding headers while keeping the leading factual answer.
    text = re.split(
        r"\s*\*\*(?:Context from Evidence|Supporting Context(?: from Evidence)?|Evidence Context)\s*:\*\*\s*",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def first_sentences(text: str, n: int) -> str:
    """Keep first n sentences with light abbreviation protection."""
    text = text.strip()
    if n <= 0 or not text:
        return text

    protected = text
    for abbr in ABBREV_FIX:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    sentences = SENT_SPLIT_RE.split(protected)
    return " ".join(sentences[:n]).replace("<DOT>", ".").strip()


def compress_answer(answer: str, max_sentences: int) -> str:
    answer = strip_markdown(answer)
    return first_sentences(answer, max_sentences) if answer else "(no answer)"


def regex_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def parse_kv_lines(body: str) -> dict:
    """Parse lines like `key: value` or `- key: value` into lowercase keys."""
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("-"):
            line = line[1:].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip()
    return out


def round2(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Old `input_text` schema support
# ---------------------------------------------------------------------------

OLD_HEADER_RE = re.compile(
    r"^(CLAIM_ID|CLAIM|EVIDENCE_QA_(\d+)|F1_NLI_(\d+)|F2_SOURCE_(\d+)|"
    r"F3_TRUST_(\d+)|F4_PROVENANCE_(\d+)):",
    re.MULTILINE,
)

OLD_FLAG_MAP = {
    "is_archive_or_copy": "archive_copy",
    "is_fact_check_article": "fact_check",
    "is_primary_or_authoritative": "primary_source",
    "is_low_credibility_or_abuse": "low_credibility",
    "is_satire_or_humor": "satire",
    "is_reference_or_data": "reference_data",
    "is_multimedia": "multimedia",
}


def split_old_sections(input_text: str) -> dict:
    matches = list(OLD_HEADER_RE.finditer(input_text))
    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(input_text)
        sections[name] = input_text[start:end].strip()
    return sections


def parse_old_qa_block(body: str) -> tuple[str, str]:
    q_match = re.search(r"^question:\s*(.*?)$", body, re.MULTILINE | re.IGNORECASE)
    question = q_match.group(1).strip() if q_match else ""
    a_match = re.search(r"^answer:\s*(.*)", body, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    answer = a_match.group(1).strip() if a_match else ""
    return question, answer


def build_old_flags(f2: dict, f4: dict) -> str:
    flags = []
    if str(f2.get("leakage_sensitive", "")).lower() == "true":
        flags.append("leakage_sensitive")
    if str(f2.get("leakage_risk", "")).lower() in {"yes", "true"}:
        flags.append("leakage_risk")
    for field, flag in OLD_FLAG_MAP.items():
        if str(f4.get(field, "")).lower() == "true":
            flags.append(flag)
    return ", ".join(flags) if flags else "-"


def render_old_evidence(idx: int, sections: dict, max_sentences: int) -> str:
    qa_body = sections.get(f"EVIDENCE_QA_{idx}", "")
    f1 = parse_kv_lines(sections.get(f"F1_NLI_{idx}", ""))
    f2 = parse_kv_lines(sections.get(f"F2_SOURCE_{idx}", ""))
    f3 = parse_kv_lines(sections.get(f"F3_TRUST_{idx}", ""))
    f4 = parse_kv_lines(sections.get(f"F4_PROVENANCE_{idx}", ""))

    question, answer = parse_old_qa_block(qa_body)
    answer = compress_answer(answer, max_sentences)

    nli = f"{f1.get('relation', f1.get('label', 'nei'))} ({round2(f1.get('confidence', '0'))})"
    src = f2.get("source_type", "unknown")
    trust = f3.get("trust_level", "unknown")
    prov = f4.get("provenance_level", "unknown")
    flags = build_old_flags(f2, f4)

    return "\n".join([
        f"[E{idx}] Q: {question if question else '(no question)'}",
        f"A: {answer}",
        f"NLI: {nli} | SRC: {src} | TRUST: {trust} | PROV: {prov} | FLAGS: {flags}",
    ])


def convert_old_input_text(input_text: str, max_sentences: int) -> str:
    sections = split_old_sections(input_text)
    claim = sections.get("CLAIM", "").strip()
    if not claim:
        raise ValueError("Could not parse CLAIM section.")

    indices = sorted(
        int(m.group(2))
        for m in OLD_HEADER_RE.finditer(input_text)
        if m.group(1).startswith("EVIDENCE_QA_")
    )
    parts = [f"CLAIM: {claim}"]
    for idx in indices:
        parts.append(render_old_evidence(idx, sections, max_sentences))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chat `messages` schema support
# ---------------------------------------------------------------------------

EVIDENCE_HEADER_RE = re.compile(r"^Evidence\s+(\d+)\s*$", re.MULTILINE | re.IGNORECASE)

CHAT_FLAG_MAP = {
    "primary_or_authoritative": "primary_source",
    "primary_or_authoritative_source": "primary_source",
    "fact_check_article": "fact_check",
    "satire_or_humor": "satire",
    "low_credibility_or_abuse": "low_credibility",
    "archive_or_copy": "archive_copy",
    "reference_or_data": "reference_data",
    "multimedia": "multimedia",
}


def extract_between(text: str, start_pat: str, end_pat: str | None = None, flags=0) -> str:
    if end_pat is None:
        pat = start_pat + r"\s*(.*)\s*$"
    else:
        pat = start_pat + r"\s*(.*?)\s*(?=" + end_pat + r")"
    m = re.search(pat, text, flags | re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_metadata_block(block: str) -> str:
    kv = parse_kv_lines(block)
    if not kv:
        return ""
    return " | ".join(f"{k}: {v}" for k, v in kv.items())


def nli_confidence(f1: dict) -> tuple[str, str]:
    label = f1.get("label") or f1.get("relation") or "nei"
    label_norm = label.lower()
    prob_key = {
        "support": "p_support",
        "supported": "p_support",
        "refute": "p_refute",
        "refuted": "p_refute",
        "nei": "p_neutral",
        "neutral": "p_neutral",
        "not enough evidence": "p_neutral",
    }.get(label_norm)

    conf = f1.get(prob_key) if prob_key else None
    if conf is None:
        probs = [f1.get("p_support"), f1.get("p_refute"), f1.get("p_neutral")]
        nums = []
        for p in probs:
            try:
                nums.append(float(p))
            except (TypeError, ValueError):
                pass
        conf = max(nums) if nums else f1.get("confidence", "0")
    return label, round2(conf)


def build_chat_flags(f2: dict, f4: dict) -> str:
    flags = []
    if str(f2.get("leakage_risk", "")).lower() in {"yes", "true", "1"}:
        flags.append("leakage_risk")

    # Important boolean provenance indicators.
    for key, flag in CHAT_FLAG_MAP.items():
        if str(f4.get(key, "")).lower() in {"yes", "true", "1"}:
            flags.append(flag)

    # Keep leakage_sensitive_source if it appears inside the comma-separated provenance_flags.
    prov_flags = f4.get("provenance_flags", "")
    prov_flag_set = {x.strip().lower() for x in prov_flags.split(",") if x.strip()}
    if "leakage_sensitive_source" in prov_flag_set and "leakage_risk" not in flags:
        flags.append("leakage_sensitive")

    # De-duplicate while preserving order.
    deduped = []
    for flag in flags:
        if flag not in deduped:
            deduped.append(flag)
    return ", ".join(deduped) if deduped else "-"


def parse_chat_evidence_block(idx: int, body: str, max_sentences: int) -> str:
    question = extract_between(body, r"^Question:\s*", r"\nAnswer:", re.IGNORECASE)
    answer = extract_between(body, r"^Answer:\s*", r"\n\nf1_relation:", re.IGNORECASE)

    f1_body = extract_between(body, r"^f1_relation:\s*", r"\n\nf2_source_quality:", re.IGNORECASE)
    f2_body = extract_between(body, r"^f2_source_quality:\s*", r"\n\nf3_trust:", re.IGNORECASE)
    f3_body = extract_between(body, r"^f3_trust:\s*", r"\n\nf4_provenance:", re.IGNORECASE)
    f4_body = extract_between(body, r"^f4_provenance:\s*", r"\n\nsource:", re.IGNORECASE)
    source_body = extract_between(body, r"^source:\s*", None, re.IGNORECASE)

    f1 = parse_kv_lines(f1_body)
    f2 = parse_kv_lines(f2_body)
    f3 = parse_kv_lines(f3_body)
    f4 = parse_kv_lines(f4_body)
    source = parse_kv_lines(source_body)

    label, conf = nli_confidence(f1)
    src = f2.get("source_type", "unknown")
    trust = f3.get("trust_level", "unknown")
    prov = f4.get("provenance_level", "unknown")
    flags = build_chat_flags(f2, f4)
    domain = source.get("domain", "")

    meta = f"NLI: {label} ({conf}) | SRC: {src}"
    if domain:
        meta += f"/{domain}"
    meta += f" | TRUST: {trust} | PROV: {prov} | FLAGS: {flags}"

    return "\n".join([
        f"[E{idx}] Q: {question if question else '(no question)'}",
        f"A: {compress_answer(answer, max_sentences)}",
        meta,
    ])


def convert_chat_user_content(content: str, max_sentences: int, keep_task: bool = True) -> str:
    claim = extract_between(
        content,
        r"^Claim:\s*",
        r"\n\n(?:Claim Metadata:|Retrieved Evidence Items:|Task:)",
        re.IGNORECASE,
    )
    if not claim:
        raise ValueError("Could not parse Claim section from chat user message.")

    metadata_block = extract_between(
        content,
        r"^Claim Metadata:\s*",
        r"\n\nRetrieved Evidence Items:",
        re.IGNORECASE,
    )
    task = extract_between(content, r"^Task:\s*", None, re.IGNORECASE)

    # Only parse evidence text between Retrieved Evidence Items and Task.
    evidence_area = extract_between(
        content,
        r"^Retrieved Evidence Items:\s*",
        r"\n\nTask:",
        re.IGNORECASE,
    )
    if not evidence_area:
        raise ValueError("Could not parse Retrieved Evidence Items section.")

    matches = list(EVIDENCE_HEADER_RE.finditer(evidence_area))
    if not matches:
        raise ValueError("Could not find any Evidence N blocks.")

    parts = [f"Claim: {claim}"]
    meta_line = parse_metadata_block(metadata_block)
    if meta_line:
        parts.append(f"Claim Metadata: {meta_line}")

    parts.append("Retrieved Evidence Items:")
    evidence_parts = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(evidence_area)
        evidence_parts.append(parse_chat_evidence_block(idx, evidence_area[start:end].strip(), max_sentences))
    parts.append("\n\n".join(evidence_parts))

    if keep_task and task:
        parts.append(f"Task:\n{task}")

    return "\n\n".join(parts)


def convert_record(rec: dict, max_sentences: int, keep_task: bool) -> tuple[dict, int, int]:
    """Return converted record plus old/new token counts for stats."""
    if "input_text" in rec:
        old_text = rec["input_text"]
        new_text = convert_old_input_text(old_text, max_sentences)
        out = dict(rec)
        out["input_text"] = new_text
        out["token_counts"] = {"regex_token_count": regex_token_count(new_text)}
        out["format_version"] = "compact_v1"
        return out, regex_token_count(old_text), regex_token_count(new_text)

    if "messages" in rec:
        out = deepcopy(rec)
        user_i = next((i for i, m in enumerate(out["messages"]) if m.get("role") == "user"), None)
        if user_i is None:
            raise ValueError("Record has messages but no user message.")
        old_text = out["messages"][user_i].get("content", "")
        new_text = convert_chat_user_content(old_text, max_sentences, keep_task=keep_task)
        out["messages"][user_i]["content"] = new_text
        return out, regex_token_count(old_text), regex_token_count(new_text)

    raise ValueError("Unsupported record schema: expected `input_text` or `messages`.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input train/dev JSONL file.")
    parser.add_argument("--output", required=True, help="Output compact JSONL path.")
    parser.add_argument("--answer-sentences", type=int, default=2,
                        help="Max sentences kept from each evidence answer. Default: 2.")
    parser.add_argument("--show-examples", type=int, default=1,
                        help="Print first N compacted user/input examples. Default: 1.")
    parser.add_argument("--drop-task", action="store_true",
                        help="For chat records, drop the final Task block from the user message.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    old_counts, new_counts = [], []
    n_records = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                out, old_n, new_n = convert_record(
                    rec,
                    max_sentences=args.answer_sentences,
                    keep_task=not args.drop_task,
                )
            except Exception as exc:
                raise ValueError(f"Line {line_no}: {exc}") from exc

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            old_counts.append(old_n)
            new_counts.append(new_n)

            if n_records < args.show_examples:
                print("=" * 70)
                print(f"EXAMPLE {n_records + 1}")
                print("=" * 70)
                if "input_text" in out:
                    print(out["input_text"])
                else:
                    user_msg = next(m for m in out["messages"] if m.get("role") == "user")
                    print(user_msg["content"])
                print()
            n_records += 1

    print("-" * 70)
    print(f"Converted {n_records} records -> {output_path}")
    if n_records:
        old_mean = statistics.mean(old_counts)
        new_mean = statistics.mean(new_counts)
        reduction = 100 * (1 - new_mean / old_mean) if old_mean else 0
        print(f"Mean regex tokens: {old_mean:.0f} -> {new_mean:.0f} ({reduction:.0f}% reduction)")
        print(f"Max regex tokens:  {max(old_counts)} -> {max(new_counts)}")


if __name__ == "__main__":
    main()
