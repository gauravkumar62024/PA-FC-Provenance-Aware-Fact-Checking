# Reproducing the primary results

`PAFC_DATA_ROOT` controls where local artifacts live; no path is hardcoded.

```bash
export PAFC_DATA_ROOT=/path/to/your/data
```

## 1. Features

```bash
# F1 - claim/evidence relation
#   cross-encoder/nli-deberta-v3-large, evidence = premise, claim = hypothesis,
#   max length 512. F1 is computed once and never recomputed between conditions.
python scripts/01_generate_f1.py --input <retrieval.json> --output-dir <f1_dir>

# F2 - source family + leakage (uses the 1,097-rule inventory in configs/f2)
python scripts/02_generate_f2.py \
    --input <f1_dir>/<split>_f1.json \
    --output-dir <f2_dir> --output-name <split>_f2.json \
    --source-list-dir configs/f2

# F3 - trust prior (derived from F2)
python scripts/03_generate_f3.py --input <f2_dir>/<split>_f2.json --output-dir <f3_dir>

# F4 - provenance / evidential role (derived from F2 + F3 + page cues)
python scripts/04_generate_f4.py --input <f3_dir>/<split>_f3.json --output-dir <f4_dir>

# QA attach - REQUIRED before SFT conversion
python scripts/05_attach_qa.py \
    --features-input <f4_dir>/<split>_f4.json --qa-input <split>.json \
    --output-dir <qa_dir> --output-name <split>_f4_qa.json
```

## 2. SFT data

```bash
python scripts/06_prepare_sft.py \
    --retrieval <qa_dir>/<split>_f4_qa.json --dev <split>.json --out <sft.jsonl>
```

## 3. Train

Shared: 4-bit QLoRA, rank 16, alpha 32, dropout 0.05, 3 epochs, 4096 context.

Batch size, accumulation and learning rate are **per dataset** — see the table in
[`PIPELINE.md`](PIPELINE.md). AVeriTeC 1e-4, RAGuard/PubHealth 2e-4, LIAR-RAW
2e-5. The shipped configs encode this; don't copy one dataset's config onto
another.

Run once per seed in {42, 191, 999}:

```bash
# AVeriTeC / PubHealth / RAGuard
llamafactory-cli train configs/train/qwen/averitec_pafc.yaml      # edit `seed:`
llamafactory-cli train configs/train/llama/averitec_pafc.yaml

# LIAR-RAW uses the REFLEX trainer
bash scripts/train_liar_raw.sh --backbone qwen --seed 42 \
     --train work/liar_raw/sft/train.jsonl --out saves/pafc/liar_raw_qwen_seed42
```

Or run the whole feature chain in one step:

```bash
bash scripts/run_pipeline.sh --dataset averitec --split train \
     --retrieval <retrieval.json> --qa <split>.json --out work/averitec
```

Peak GPU is roughly 18.5–24 GB depending on dataset. If a run OOMs, wait for
headroom rather than lowering batch size or context — changing either breaks
comparability with the published configuration.

## 4. Infer

```bash
python scripts/infer_averitec.py  --config <yaml> --test_file <eval.json> \
    --adapter_dir <adapter> --out_dir <preds>
python scripts/infer_raguard.py   --base_model <hf-id> --adapter_dir <adapter> \
    --test_file <eval.json> --output_jsonl <preds>/test_predictions.jsonl
python scripts/infer_pubhealth.py --config <yaml> --test_file <eval.json> \
    --pred_dir <preds> --load_in_4bit
python scripts/infer_liar_raw.py  --base_model <hf-id> --adapter_path <adapter> \
    --test_file <eval.json> --output_dir <preds> --expected_count 1251
```

## 5. Evaluate

```bash
python scripts/evaluate.py --predictions <preds>/predictions.jsonl

cp configs/runs.example.json configs/runs.json   # edit paths
python scripts/report_results.py --runs configs/runs.json --csv results.csv
```

`report_results.py` recomputes every cell from predictions and prints the seed
count behind each one. A cell backed by fewer than three seeds is marked `partial`
rather than being presented as a three-seed mean.
