#!/usr/bin/env bash
# Wire the PA-FC datasets and configs into an existing LLaMA-Factory checkout.
#
#   bash llamafactory/setup.sh /path/to/LLaMA-Factory
#
# Copies the three LLaMA-Factory datasets into <LF>/data/, merges our entries
# into <LF>/data/dataset_info.json (keeping everything already there), and
# copies the training configs into <LF>/examples/train_lora/pafc/.
set -euo pipefail

LF="${1:-}"
if [[ -z "$LF" ]]; then
  echo "usage: bash llamafactory/setup.sh /path/to/LLaMA-Factory" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -d "$LF/data" ]] || { echo "not a LLaMA-Factory checkout (no data/): $LF" >&2; exit 1; }

# 1. Datasets. Two datasets are deliberately absent: LIAR-RAW trains under
#    REFLEX, and PubHealth trains under the TakeItEasy CausalLM pipeline
#    (see takeiteasy/README.md). Only AVeriTeC and RAGuard use LLaMA-Factory.
echo "[1/3] copying datasets -> $LF/data/"
copy_one() {  # <gz path> <target name>
  local gz="$HERE/$1" out="$LF/data/$2"
  [[ -f "$gz" ]] || { echo "  missing $gz (run scripts/prepare_data.sh first?)" >&2; return 1; }
  gunzip -c "$gz" > "$out"
  printf '  %-28s %s\n' "$2" "$(du -h "$out" | cut -f1)"
}
copy_one data/averitec/train.json.gz  pafc_averitec_train.json
copy_one data/averitec/eval.json.gz   pafc_averitec_eval.json
copy_one data/raguard/train.json.gz   pafc_raguard_train.json
copy_one data/raguard/eval.json.gz    pafc_raguard_eval.json
copy_one data/raguard/test.json.gz    pafc_raguard_test.json

# 2. Registration. Merge, never overwrite: a LLaMA-Factory checkout ships
#    hundreds of its own entries and clobbering them breaks every other recipe.
echo "[2/3] merging dataset_info.json"
python3 - "$HERE/llamafactory/dataset_info.json" "$LF/data/dataset_info.json" <<'PY'
import json, shutil, sys
ours_path, theirs_path = sys.argv[1], sys.argv[2]
ours = json.load(open(ours_path))
try:
    theirs = json.load(open(theirs_path))
except FileNotFoundError:
    theirs = {}
else:
    shutil.copy(theirs_path, theirs_path + ".bak")
    print(f"  backup -> {theirs_path}.bak")
added = [k for k in ours if k not in theirs]
replaced = [k for k in ours if k in theirs and theirs[k] != ours[k]]
theirs.update(ours)
json.dump(theirs, open(theirs_path, "w"), indent=2, ensure_ascii=False)
print(f"  {len(added)} added, {len(replaced)} updated, {len(theirs)} entries total")
PY

# 3. Configs.
echo "[3/3] copying configs -> $LF/examples/train_lora/pafc/"
mkdir -p "$LF/examples/train_lora/pafc"
cp -r "$HERE/configs/train/qwen" "$HERE/configs/train/llama" "$LF/examples/train_lora/pafc/"

cat <<EOF

Done. Train with:

  cd $LF
  llamafactory-cli train examples/train_lora/pafc/qwen/averitec_pafc.yaml

PubHealth and LIAR-RAW do not go through LLaMA-Factory:

  bash takeiteasy/run_pubhealth_causal_seed.sh      # PubHealth, see takeiteasy/README.md


  bash scripts/train_liar_raw.sh --backbone qwen --seed 42 \\
       --train data/liar_raw/train.json --out saves/pafc/liar_raw_qwen_seed42
EOF
