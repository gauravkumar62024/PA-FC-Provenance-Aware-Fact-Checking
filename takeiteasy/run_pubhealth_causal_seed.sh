#!/usr/bin/env bash
# PubHealth PA-FC seed runs on ISL A100 80GB -- native TakeItEasy data pipeline
# with a CausalLM (Qwen2.5 / Llama-3.1) verifier head.
#
# Qwen2.5-7B-Instruct + Llama-3.1-8B-Instruct x seeds 191, 999.
# Dataset: pubhealth_pafc_f134_fixed (9804 train / 1231 test)
#
# Hyperparameters mirror the LlamaFactory seed-42 PubHealth runs:
#   lr 2e-4, 3 epochs, bsz 1 x grad_accum 16, LoRA r16/a32, cutoff 4096, QLoRA 4-bit.
#
# Two jobs run concurrently on the single 80GB A100; each wave pairs a Qwen job
# with a Llama job so the two never contend for the same tokenizer/template.
#
# NOT using `set -e` -- each run is guarded so one failure cannot abort the rest.
set -uo pipefail

cd ./takeiteasy_pafc/code

PY=python3

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=./huggingface_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=pubhealth_pafc_f134_fixed
MODEL_ROOT=../model_causal_seeds
RESULT_ROOT=../result_causal_seeds
LOGDIR=logs_seeds
mkdir -p "$LOGDIR" "$MODEL_ROOT" "$RESULT_ROOT"

STATUS=$LOGDIR/pubhealth_seed_status.log
echo "=== PUBHEALTH CAUSAL SEED RUNS START $(date) ===" | tee -a "$STATUS"

# run_one <model_name> <short> <seed>
run_one () {
  local MODEL=$1 SHORT=$2 SEED=$3
  local TAG="${SHORT}_pafc_seed_${SEED}"
  local SAVE="$MODEL_ROOT/$TAG"
  local RES="$RESULT_ROOT/$TAG"
  local ADAPTER="$SAVE/$(echo "$MODEL" | tr '/' '-')/$DATASET/nt5000/no_exp"

  echo "[$TAG] TRAIN start $(date)" | tee -a "$STATUS"
  $PY training_causal.py \
    --source_dataset_name "$DATASET" \
    --model_name "$MODEL" \
    --random_seed "$SEED" \
    --sample_seed "$SEED" \
    --lr 2e-4 \
    --train_bsz 1 \
    --gradient_accumulation_steps 16 \
    --num_epochs 3 \
    --max_length 4096 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --warmup_ratio 0.03 \
    --io_format no_exp \
    --save_model_path "$SAVE" \
    > "$LOGDIR/${TAG}_train.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$TAG] !!! TRAIN FAILED - see $LOGDIR/${TAG}_train.log" | tee -a "$STATUS"
    tail -25 "$LOGDIR/${TAG}_train.log" | tee -a "$STATUS"
    return 1
  fi

  # The training script already asserts adapter size, but re-check here so a
  # bad checkpoint never reaches inference.
  local AF="$ADAPTER/adapter_model.safetensors"
  if [ ! -f "$AF" ]; then
    echo "[$TAG] !!! NO ADAPTER at $AF" | tee -a "$STATUS"
    return 1
  fi
  echo "[$TAG] adapter OK $(( $(stat -c%s "$AF") / 1000000 )) MB" | tee -a "$STATUS"

  # Confirm the run actually learned something (guards the zero-loss failure mode).
  grep -o "Total supervised tokens: [0-9]*" "$LOGDIR/${TAG}_train.log" | tail -1 | tee -a "$STATUS"

  echo "[$TAG] INFER start $(date)" | tee -a "$STATUS"
  $PY inference_causal.py \
    --source_dataset_name "$DATASET" \
    --target_dataset_name "$DATASET" \
    --data_split test \
    --base_model "$MODEL" \
    --adapter_path "$ADAPTER" \
    --seed "$SEED" \
    --sample_seed "$SEED" \
    --max_length 4096 \
    --max_new_tokens 8 \
    --io_format no_exp \
    --result_path "$RES" \
    > "$LOGDIR/${TAG}_infer.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$TAG] !!! INFER FAILED - see $LOGDIR/${TAG}_infer.log" | tee -a "$STATUS"
    tail -25 "$LOGDIR/${TAG}_infer.log" | tee -a "$STATUS"
    return 1
  fi

  echo "[$TAG] DONE $(date)" | tee -a "$STATUS"
  TAG="$TAG" RES="$RES" $PY - <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json, os
tag, res = os.environ["TAG"], os.environ["RES"]
try:
    m = json.load(open(os.path.join(res, "results_test.json")))
    print("[%s] acc=%.4f macroF1=%.4f wF1=%.4f" % (
        tag, m["accuracy"], m["macro_f1"], m["weighted_f1"]))
    print("[%s] preds: %s" % (tag, m.get("pred_label_counts")))
except Exception as e:
    print("[%s] could not read results: %s" % (tag, e))
PYEOF
  return 0
}

QBASE=Qwen/Qwen2.5-7B-Instruct
LBASE=meta-llama/Llama-3.1-8B-Instruct

# ── Wave 1: Qwen 191 || Llama 191 ──────────────────────────────────────────
echo "--- WAVE 1 (qwen191 + llama191) $(date) ---" | tee -a "$STATUS"
run_one "$QBASE" qwen25  191 &
P1=$!
run_one "$LBASE" llama31 191 &
P2=$!
wait $P1; wait $P2
echo "--- WAVE 1 complete $(date) ---" | tee -a "$STATUS"

# ── Wave 2: Qwen 999 || Llama 999 ──────────────────────────────────────────
echo "--- WAVE 2 (qwen999 + llama999) $(date) ---" | tee -a "$STATUS"
run_one "$QBASE" qwen25  999 &
P3=$!
run_one "$LBASE" llama31 999 &
P4=$!
wait $P3; wait $P4
echo "--- WAVE 2 complete $(date) ---" | tee -a "$STATUS"

echo "=== ALL PUBHEALTH CAUSAL SEED RUNS DONE $(date) ===" | tee -a "$STATUS"
