#!/usr/bin/env bash
# End-to-end PA-FC pipeline for one dataset split:
#   F1 -> F2 -> F3 -> F4 -> QA -> training text -> labels -> COMPACT -> LLaMA-Factory.
#
#   bash scripts/run_pipeline.sh --dataset averitec --split train \
#        --retrieval /path/to/retrieval.json --qa /path/to/train.json --out work/averitec
#
# Produces <out>/sft/<split>.json (Alpaca array) plus a dataset_info snippet,
# ready to register with LLaMA-Factory and train.
set -euo pipefail

DATASET=""; SPLIT=""; RETRIEVAL=""; QA=""; OUT=""; TOPK=10
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)   DATASET="$2"; shift 2 ;;
    --split)     SPLIT="$2"; shift 2 ;;
    --retrieval) RETRIEVAL="$2"; shift 2 ;;
    --qa)        QA="$2"; shift 2 ;;
    --out)       OUT="$2"; shift 2 ;;
    --top-k)     TOPK="$2"; shift 2 ;;
    -h|--help)   sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
for v in DATASET SPLIT RETRIEVAL OUT; do
  [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }
done
[[ -f "$RETRIEVAL" ]] || { echo "retrieval file not found: $RETRIEVAL" >&2; exit 1; }

mkdir -p "$OUT"/{f1,f2,f3,f4,qa,text,compact,sft}
echo "[1/9] F1  relation"
$PY "$HERE/scripts/01_generate_f1.py" --input "$RETRIEVAL" --output-dir "$OUT/f1"
F1=$(ls "$OUT/f1"/*.json | head -1)

echo "[2/9] F2  source family (1,097-rule inventory)"
$PY "$HERE/scripts/02_generate_f2.py" --input "$F1" \
    --output-dir "$OUT/f2" --output-name "${SPLIT}_f2.json" \
    --source-list-dir "$HERE/configs/f2"

echo "[3/9] F3  trust prior"
$PY "$HERE/scripts/03_generate_f3.py" --input "$OUT/f2/${SPLIT}_f2.json" \
    --output-dir "$OUT/f3" --output-name "${SPLIT}_f3.json"

echo "[4/9] F4  provenance / evidential role"
$PY "$HERE/scripts/04_generate_f4.py" --input "$OUT/f3/${SPLIT}_f3.json" \
    --output-dir "$OUT/f4" --output-name "${SPLIT}_f4.json"

if [[ -n "$QA" ]]; then
  echo "[5/9] QA  attach"
  $PY "$HERE/scripts/05_attach_qa.py" --features-input "$OUT/f4/${SPLIT}_f4.json" \
      --qa-input "$QA" --output-dir "$OUT/qa" --output-name "${SPLIT}_f4_qa.json"
  FEAT="$OUT/qa/${SPLIT}_f4_qa.json"
else
  echo "[5/9] QA  skipped (no --qa given)"
  FEAT="$OUT/f4/${SPLIT}_f4.json"
fi

echo "[6/9] training text (verbose feature blocks, top-k)"
$PY "$HERE/scripts/06_build_training_text.py" --input "$FEAT" \
    --output "$OUT/text/${SPLIT}.jsonl" --top_k "$TOPK"

echo "[7/9] attach gold labels"
$PY "$HERE/scripts/07_attach_labels.py" --converted "$OUT/text/${SPLIT}.jsonl" \
    --original "${QA:-$RETRIEVAL}" --output "$OUT/text/${SPLIT}_labeled.jsonl" \
    --unmatched_output "$OUT/text/${SPLIT}_unmatched.jsonl"
UNMATCHED=$(wc -l < "$OUT/text/${SPLIT}_unmatched.jsonl" 2>/dev/null || echo 0)
if [[ "$UNMATCHED" -gt 0 ]]; then
  echo "  WARNING: $UNMATCHED claims got no gold label -> $OUT/text/${SPLIT}_unmatched.jsonl"
  echo "  Those rows carry unreliable labels. Expected count is 0."
fi

# The verifier was trained on the COMPACT one-line form, not the verbose blocks.
# Skipping this stage produces prompts that do not match training.
echo "[8/9] compact metadata line"
$PY "$HERE/scripts/08_render_compact.py" \
    --input "$OUT/text/${SPLIT}_labeled.jsonl" \
    --output "$OUT/compact/${SPLIT}.jsonl" --answer-sentences 2 --show-examples 0

echo "[9/9] LLaMA-Factory format (Alpaca array + dataset_info snippet)"
$PY "$HERE/scripts/09_prepare_llamafactory.py" --input "$OUT/compact/${SPLIT}.jsonl" \
    --output "$OUT/sft/${SPLIT}.json" --dataset-name "pafc_${DATASET}_${SPLIT}" \
    --style alpaca --output-label-style label --oversample-max-ratio 1.0

echo
echo "done -> $OUT/sft/${SPLIT}.json"
echo "register: merge $OUT/sft/${SPLIT}.dataset_info.json into LLaMA-Factory data/dataset_info.json"
echo "next:     llamafactory-cli train configs/train/<backbone>/${DATASET}_pafc.yaml"
