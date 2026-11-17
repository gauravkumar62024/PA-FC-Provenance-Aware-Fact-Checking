#!/usr/bin/env python3
"""Render the -40-rules SFT file by rewriting ONLY the metadata line.

The shipped SFT (pafc_top10_train.json) was not produced by the converter in
PA-FC/dev, so it cannot be rebuilt from scratch faithfully. Instead we take it
as the base and substitute SRC / TRUST / PROV / FLAGS from a features file,
leaving NLI, claim text, evidence text and evidence order byte-identical.

Self-validating: run with --features <ORIGINAL features> and it must reproduce
the base exactly. Only then is the -40 rendering trustworthy.

  python render_minus40_sft.py --base <sft.json> --features <f4_qa.json> \
      --out <out.json> [--validate]
"""
import argparse, json, re, sys, collections

META = re.compile(r"^NLI: .*?\| SRC: .*?\| TRUST: .*?\| PROV: .*?\| FLAGS:.*$", re.M)
CLAIM = re.compile(r"CLAIM:\s*(.*?)(?:\n\n|\n\[E)", re.S | re.I)

# order matters: it must reproduce the original strings exactly
FLAG_FIELDS = [
    ("f2_leakage_sensitive", "leakage_sensitive"),
    ("f4_is_archive_or_copy", "archive_copy"),
    ("f4_is_fact_check_article", "fact_check"),
    ("f4_is_primary_or_authoritative", "primary_source"),
    ("f4_is_low_credibility_or_abuse", "low_credibility"),
    ("f4_is_satire_or_humor", "satire"),
    ("f4_is_reference_or_data", "reference_data"),
    ("f4_is_multimedia", "multimedia"),
]


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def load_any(p):
    t = open(p, encoding="utf-8").read().strip()
    return json.loads(t) if t.startswith("[") else [json.loads(l) for l in t.splitlines() if l.strip()]


def build_index(feats):
    """claim -> LIST of candidate evidence lists.

    40 claims recur in the features file with different evidence, so a single
    entry per claim silently mis-assigns them. Keep every candidate and let the
    caller disambiguate on the NLI sequence, which is identical across the two
    conditions because F1 is never recomputed.
    """
    idx = collections.defaultdict(list)
    for r in feats:
        ev = r.get("top_10")
        if ev is None:
            ev = next((v for v in r.values() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
        if ev:
            idx[norm(r.get("claim", ""))].append(ev)
    return idx


def nli_sig(lines):
    """('nei', '1.00') per metadata line -- the disambiguation key."""
    out = []
    for l in lines:
        m = re.match(r"^NLI: (\S+) \(([\d.]+)\)", l)
        out.append((m.group(1), m.group(2)) if m else None)
    return out


def pick(cands, base_lines):
    """Choose the candidate whose F1 relation/confidence sequence matches."""
    if len(cands) == 1:
        return cands[0]
    want = nli_sig(base_lines)
    best, score = None, -1
    for ev in cands:
        got = [(str(e.get("f1_relation_label")), f"{float(e.get('f1_confidence', 0)):.2f}")
               for e in ev]
        s = sum(1 for x, y in zip(want, got) if x == y)
        if s > score:
            best, score = ev, s
    # Only accept an exact F1 match. If no candidate matches perfectly the record
    # is ambiguous, so return None and leave its metadata line untouched -- that
    # keeps it IDENTICAL across both conditions, contributing nothing to the delta.
    if score < len(want):
        return None
    return best


def flags_for(e):
    on = [label for key, label in FLAG_FIELDS if e.get(key)]
    return ", ".join(on) if on else "-"


def line_for(e, original_line):
    m = re.match(r"^(NLI: [^|]*)\|", original_line)
    nli = m.group(1).strip() if m else "NLI: nei (0.00)"
    return (f"{nli} | SRC: {e.get('f2_source_type') or 'unknown'}"
            f" | TRUST: {e.get('f3_trust_level') or 'unknown'}"
            f" | PROV: {e.get('f4_provenance_level') or 'unknown'}"
            f" | FLAGS: {flags_for(e)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true",
                    help="assert the render reproduces --base exactly")
    a = ap.parse_args()

    base = load_any(a.base)
    idx = build_index(load_any(a.features))
    out = []
    miss = changed = lines_total = lines_changed = 0
    ev_short = 0

    for rec in base:
        txt = rec.get("input", "")
        cm = CLAIM.search(txt)
        cands = idx.get(norm(cm.group(1))) if cm else None
        ev = pick(cands, META.findall(txt)) if cands else None
        if ev is None:
            miss += 1
            out.append(rec)
            continue
        n = [0]

        def sub(m):
            i = n[0]; n[0] += 1
            nonlocal lines_total, lines_changed, ev_short
            lines_total += 1
            if i >= len(ev):
                ev_short += 1
                return m.group(0)
            new = line_for(ev[i], m.group(0))
            if new != m.group(0):
                lines_changed += 1
            return new

        new_txt = META.sub(sub, txt)
        if new_txt != txt:
            changed += 1
        nr = dict(rec); nr["input"] = new_txt
        out.append(nr)

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  base records        : {len(base)}")
    print(f"  claim match failures: {miss}")
    print(f"  evidence-index short: {ev_short}")
    print(f"  metadata lines seen : {lines_total}")
    print(f"  metadata lines changed: {lines_changed}")
    print(f"  records changed     : {changed}")
    print(f"  -> {a.out}")

    if a.validate:
        ok = (lines_changed == 0 and miss == 0 and ev_short == 0)
        print(f"  VALIDATION: {'PASS - render reproduces the base exactly' if ok else 'FAIL - render differs from base'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
