#!/usr/bin/env python3
"""Build the PA-FC results table from prediction files.

Every number in the table is recomputed here from per-claim predictions. No
value is hard-coded from the paper.

Each cell is the mean over seeds {42, 191, 999}. The script reports how many
seeds it actually found, so a partial mean is never silently presented as a
three-seed mean.

    python scripts/report_results.py --runs configs/runs.json [--csv results.csv]

`runs.json` maps each cell to one prediction directory per seed; see
`configs/runs.example.json`. Paths are local to whoever runs this, which
is why they live in a config file rather than in the code.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from statistics import mean, pstdev

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from pafc import macro_f1, accuracy  # noqa: E402

# (gold_key, pred_key) candidates, in the order the four pipelines write them.
SCHEMAS = [
    ("predictions.jsonl", "gold", "prediction"),        # AVeriTeC
    ("test_predictions.jsonl", "gold", "pred"),         # RAGuard
    ("predictions.jsonl", "gold_label", "predicted_label"),  # LIAR-RAW / REFLEX
]


def load_pairs(d):
    """(gold, pred) pairs from a prediction directory, whichever schema it uses."""
    for name, gk, pk in SCHEMAS:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        out = []
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if gk in r and pk in r:
                out.append((str(r[gk]).strip(), str(r[pk]).strip()))
        if out:
            return out
    p = os.path.join(d, "predictions.csv")                # PubHealth
    if os.path.exists(p):
        return [(str(r["gold"]).strip(), str(r["pred"]).strip())
                for r in csv.DictReader(open(p, encoding="utf-8"))]
    p = os.path.join(d, "report_test.json")               # TakeItEasy summary
    if os.path.exists(p):
        return None                                       # handled by caller
    return None


def score(d):
    pairs = load_pairs(d)
    if pairs:
        return len(pairs), macro_f1(pairs), accuracy(pairs)
    p = os.path.join(d, "report_test.json")
    if os.path.exists(p):
        r = json.load(open(p))
        n = int(r.get("macro avg", {}).get("support", 0)) or None
        return n, r["macro avg"]["f1-score"], r["accuracy"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="JSON mapping cells -> prediction dirs")
    ap.add_argument("--csv", help="also write the table as CSV")
    a = ap.parse_args()

    spec = json.load(open(a.runs))
    rows = []
    print(f"{'Dataset':<11}{'System':<24}{'n':>6}{'Macro-F1':>10}{'Acc.':>8}{'seeds':>7}")
    print("-" * 66)
    for dataset, systems in spec.items():
        if dataset.startswith("_"):   # comment keys
            continue
        for system, dirs in systems.items():
            if system.startswith("_"):
                continue
            if isinstance(dirs, str):
                dirs = [dirs]
            vals = [v for v in (score(d) for d in dirs if os.path.isdir(d)) if v]
            if not vals:
                print(f"{dataset:<11}{system:<24}{'--':>6}{'MISSING':>10}")
                rows.append(dict(dataset=dataset, system=system, n="", macro_f1="",
                                 accuracy="", n_seeds=0, status="MISSING"))
                continue
            n = vals[0][0]
            f1 = mean(v[1] for v in vals)
            ac = mean(v[2] for v in vals)
            sd = pstdev([v[1] for v in vals]) if len(vals) > 1 else 0.0
            partial_row = len(vals) < 3
            flag = "  <- partial" if partial_row else ""
            print(f"{dataset:<11}{system:<24}{n:>6}{f1:>10.4f}{ac:>8.4f}"
                  f"{len(vals):>7}{flag}")
            rows.append(dict(dataset=dataset, system=system, n=n,
                             macro_f1=round(f1, 4), accuracy=round(ac, 4),
                             macro_f1_sd=round(sd, 4), n_seeds=len(vals),
                             status="partial" if partial_row else "ok"))

    partial = [r for r in rows if r["status"] == "partial"]
    missing = [r for r in rows if r["status"] == "MISSING"]
    if partial:
        print(f"\n{len(partial)} cell(s) averaged over fewer than 3 seeds — "
              "these are not three-seed means.")
    if missing:
        print(f"{len(missing)} cell(s) had no prediction directory.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
