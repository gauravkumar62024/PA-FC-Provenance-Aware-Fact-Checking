#!/usr/bin/env python3
"""Recompute metrics from a prediction file. No metric is ever copied from text.

    python scripts/10_evaluate.py --predictions <file> [--gold-key gold] [--pred-key prediction]

Handles the four prediction schemas used across this project:
  {id, gold, prediction}            AVeriTeC
  {idx, gold, pred}                 RAGuard
  {id, gold_label, predicted_label} LIAR-RAW / REFLEX
  gold,pred CSV                     PubHealth
"""
import argparse, csv, json, os, sys
from collections import Counter

def load(path, gold_key=None, pred_key=None):
    rows = []
    if path.endswith(".csv"):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rows.append((str(r.get(gold_key or "gold")).strip(),
                         str(r.get(pred_key or "pred")).strip()))
        return rows
    GK = [gold_key] if gold_key else ["gold", "gold_label", "label"]
    PK = [pred_key] if pred_key else ["prediction", "pred", "predicted_label"]
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        g = next((d[k] for k in GK if k in d), None)
        p = next((d[k] for k in PK if k in d), None)
        if g is None or p is None:
            continue
        rows.append((str(g).strip(), str(p).strip()))
    return rows

def metrics(pairs):
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    per, macro, n_lab = {}, 0.0, 0
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp + fn == 0:
            continue                      # label absent from gold: not a real class
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = dict(precision=round(prec, 4), recall=round(rec, 4),
                        f1=round(f1, 4), support=tp + fn)
        macro += f1; n_lab += 1
    acc = sum(1 for g, p in pairs if g == p) / len(pairs)
    total = sum(v["support"] for v in per.values())
    wf1 = sum(v["f1"] * v["support"] for v in per.values()) / total if total else 0.0
    return dict(n=len(pairs), accuracy=round(acc, 4),
                macro_f1=round(macro / n_lab if n_lab else 0.0, 4),
                weighted_f1=round(wf1, 4), per_class=per)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold-key"); ap.add_argument("--pred-key")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    pairs = load(a.predictions, a.gold_key, a.pred_key)
    if not pairs:
        sys.exit(f"no (gold, prediction) pairs parsed from {a.predictions}")
    m = metrics(pairs)
    print(f"file        : {a.predictions}")
    print(f"n           : {m['n']}")
    print(f"accuracy    : {m['accuracy']:.4f}")
    print(f"macro F1    : {m['macro_f1']:.4f}")
    print(f"weighted F1 : {m['weighted_f1']:.4f}")
    print("per class   :")
    for lab, v in m["per_class"].items():
        print(f"  {lab:<24} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} n={v['support']}")
    print("confusion (gold -> pred):")
    for (g, p), c in Counter(pairs).most_common():
        if g != p:
            print(f"  {g:<22} -> {p:<22} {c}")
    if a.json_out:
        json.dump(m, open(a.json_out, "w"), indent=2)
        print("wrote", a.json_out)

if __name__ == "__main__":
    main()
