#!/usr/bin/env bash
# Train one PA-FC cell at every seed the paper reports: {42, 191, 999}.
#
#   bash scripts/train_all_seeds.sh --config configs/train/qwen/averitec_pafc.yaml
#   bash scripts/train_all_seeds.sh --config configs/train/llama/raguard_pafc.yaml --seeds "42 191"
#
# Writes one adapter per seed to <output_dir>_seed<N> and leaves the shipped
# config untouched (a per-seed copy is written under work/configs/).
set -euo pipefail
CONFIG=""; SEEDS="42 191 999"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --seeds)  SEEDS="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -f "$CONFIG" ]] || { echo "need --config <existing yaml>" >&2; exit 2; }

BASE_OUT=$(grep -E '^output_dir:' "$CONFIG" | head -1 | sed 's/^output_dir:[[:space:]]*//')
BASE_OUT=${BASE_OUT%_seed42}
NAME=$(basename "$CONFIG" .yaml)
mkdir -p "$HERE/work/configs"

for S in $SEEDS; do
  CFG="$HERE/work/configs/${NAME}_seed${S}.yaml"
  sed -e "s|^seed:.*|seed: ${S}|" \
      -e "s|^output_dir:.*|output_dir: ${BASE_OUT}_seed${S}|" "$CONFIG" > "$CFG"
  echo "=== seed ${S} -> ${BASE_OUT}_seed${S} ==="
  llamafactory-cli train "$CFG"
done

echo
echo "done. All seeds for ${NAME}:"
for S in $SEEDS; do echo "  ${BASE_OUT}_seed${S}"; done
echo "Average them with: python scripts/report_results.py --runs configs/runs.json"
