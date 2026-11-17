#!/usr/bin/env python3
"""Put every LIAR-RAW track on the same 3-way label space.

The HerO2 tracks predict the original 6-way LIAR labels; REFLEX predicts 3-way
directly. To compare them, the 6-way outputs are collapsed with the standard
LIAR 3-way mapping, applied identically to gold and prediction:

    pants-fire, false, barely-true  -> FALSE
    half-true                       -> HALF-TRUE
    mostly-true, true               -> TRUE
"""

import os as _os
# Root of the local artifact tree. Override with PAFC_DATA_ROOT so the
# repository carries no machine-specific absolute paths.
PAFC_DATA_ROOT = _os.environ.get("PAFC_DATA_ROOT", _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data"))

import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

SIX_TO_THREE = {
    "pants-fire": "FALSE", "false": "FALSE", "barely-true": "FALSE",
    "half-true": "HALF-TRUE",
    "mostly-true": "TRUE", "true": "TRUE",
}
LABELS = ["TRUE", "HALF-TRUE", "FALSE"]


def to3(x):
    s = str(x).strip().lower()
    if s in SIX_TO_THREE:
        return SIX_TO_THREE[s]
    u = str(x).strip().upper()
    return u if u in LABELS else "FALSE"   # unparsable -> majority class


def load(path, gold_key, pred_key, collapse):
    g, p = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gv, pv = d.get(gold_key), d.get(pred_key)
        if gv is None or pv is None:
            continue
        g.append(to3(gv) if collapse else str(gv).strip().upper())
        p.append(to3(pv) if collapse else str(pv).strip().upper())
    return g, p


TRACKS = [
    ("HerO2 baseline (6->3)",
     "" + PAFC_DATA_ROOT + "/liar_raw/data_store/hero2_lair_raw/test_veracity_prediction.eval.jsonl",
     "gold", "pred", True),
    ("HerO2 + PA-FC (6->3)",
     "" + PAFC_DATA_ROOT + "/llamafactory/saves/lair_raw_pafc_qwen25_7b_qlora/merged_bf16_test_predictions/predictions.jsonl",
     "gold", "pred", True),
    ("REFLEX baseline (native 3)",
     "" + PAFC_DATA_ROOT + "/liar_raw/reflex_baseline.jsonl",
     "gold_label", "predicted_label", False),
    ("REFLEX + PA-FC (native 3)",
     "" + PAFC_DATA_ROOT + "/liar_raw/reflex_pafc.jsonl",
     "gold_label", "predicted_label", False),
]

OUT = Path("" + PAFC_DATA_ROOT + "/llamafactory/wsdm_experiments/metrics")
OUT.mkdir(parents=True, exist_ok=True)

summary = []
full = []

for name, path, gk, pk, collapse in TRACKS:
    if not Path(path).exists():
        print(f"MISSING: {name} -> {path}")
        continue
    g, p = load(path, gk, pk, collapse)
    rep = classification_report(g, p, labels=LABELS, digits=4,
                                output_dict=True, zero_division=0)
    txt = classification_report(g, p, labels=LABELS, digits=4, zero_division=0)
    cm = confusion_matrix(g, p, labels=LABELS)

    full.append(f"{'='*70}\n{name}   (n={len(g)})\n{'='*70}\n{txt}\n"
                f"confusion matrix (rows=gold {LABELS}):\n{cm}\n")
    summary.append({
        "track": name, "n": len(g),
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        **{f"F1_{l}": rep[l]["f1-score"] for l in LABELS},
    })

# ── printed table ────────────────────────────────────────────────────────
hdr = (f"{'Track':<30}{'n':>6}{'Acc':>9}{'MacroF1':>10}{'WtdF1':>9}"
       f"{'F1 TRUE':>10}{'F1 HALF':>10}{'F1 FALSE':>10}")
print("\nLIAR-RAW — all tracks on the 3-way label space\n")
print(hdr)
print("-" * len(hdr))
for r in summary:
    print(f"{r['track']:<30}{r['n']:>6}{r['accuracy']:>9.4f}{r['macro_f1']:>10.4f}"
          f"{r['weighted_f1']:>9.4f}{r['F1_TRUE']:>10.4f}{r['F1_HALF-TRUE']:>10.4f}"
          f"{r['F1_FALSE']:>10.4f}")

(OUT / "liar_raw_3way_reports.txt").write_text("\n".join(full), encoding="utf-8")
import csv
with open(OUT / "liar_raw_3way_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"\nwrote {OUT}/liar_raw_3way_reports.txt and liar_raw_3way_summary.csv")
