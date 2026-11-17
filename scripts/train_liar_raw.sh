#!/usr/bin/env bash
# PA-FC on LIAR-RAW. This dataset uses the REFLEX trainer rather than
# LLaMA-Factory, so it gets its own entry point with the same hyperparameters.
#
#   bash scripts/train_liar_raw.sh --backbone qwen --seed 42 \
#        --train work/liar_raw/sft/train.jsonl --val work/liar_raw/sft/val.jsonl \
#        --out saves/pafc/liar_raw_qwen_seed42
set -euo pipefail

# LIAR-RAW trains with the REFLEX trainer, which needs REFLEX's `llmzoo` package.
# It is NOT vendored here: upstream REFLEX ships no license, so redistributing it
# is not ours to do. Clone it and put it on PYTHONPATH.
if ! python3 -c "import llmzoo" 2>/dev/null; then
  cat >&2 <<'MSG'
ERROR: python package `llmzoo` not found. The REFLEX trainer needs it.

  git clone https://github.com/renatz/REFLEX.git
  export PYTHONPATH=$PWD/REFLEX:$PYTHONPATH

Then re-run this script. See docs/PIPELINE.md (LIAR-RAW).
MSG
  exit 1
fi

BACKBONE=qwen; SEED=42; TRAIN=""; VAL=""; OUT=""
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) BACKBONE="$2"; shift 2 ;;
    --seed)     SEED="$2"; shift 2 ;;
    --train)    TRAIN="$2"; shift 2 ;;
    --val)      VAL="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$BACKBONE" in
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  *) echo "--backbone must be qwen or llama" >&2; exit 2 ;;
esac
for v in TRAIN OUT; do [[ -n "${!v}" ]] || { echo "missing --${v,,}" >&2; exit 2; }; done

$PY "$HERE/scripts/train_reflex.py" \
  --model_name_or_path "$MODEL" --task_template liar_raw_cev \
  --lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --train_data_path "$TRAIN" ${VAL:+--eval_data_path "$VAL"} \
  --output_dir "$OUT" --model_max_length 4096 --num_train_epochs 3 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 96 --eval_strategy epoch \
  --prediction_loss_only True --eval_accumulation_steps 8 --save_strategy no \
  --logging_strategy steps --logging_steps 10 --learning_rate 2e-5 \
  --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --max_grad_norm 1.0 --bf16 True --tf32 True --gradient_checkpointing True \
  --dataloader_num_workers 0 --seed "$SEED" --data_seed "$SEED" \
  --report_to none --remove_unused_columns False
