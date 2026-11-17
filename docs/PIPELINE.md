# The PA-FC pipeline, end to end

```
AVeriTeC / PubHealth / RAGuard / LIAR-RAW
  │
  ├─ 0  retrieval             NOT IN THIS REPO — see "Stage 0" below
  │                           produces <split>_retrieval.json
  │
  ├─ 1  F1  relation          scripts/01_generate_f1.py      NLI, per evidence item
  ├─ 2  F2  source family     scripts/02_generate_f2.py      1,097-rule inventory
  ├─ 3  F3  trust prior       scripts/03_generate_f3.py      derived from F2
  ├─ 4  F4  provenance role   scripts/04_generate_f4.py      derived from F2+F3+page cues
  ├─ 5  QA attach             scripts/05_attach_qa.py        REQUIRED before stage 6
  │
  ├─ 6  training text         scripts/06_build_training_text.py   verbose blocks, top-k
  ├─ 7  gold labels           scripts/07_attach_labels.py         match by normalized claim
  ├─ 8  COMPACT metadata line scripts/08_render_compact.py  the form trained on
  ├─ 9  LLaMA-Factory format  scripts/09_prepare_llamafactory.py   Alpaca array + dataset_info
  │
  ├─ 10 QLoRA SFT             llamafactory-cli train  ... or scripts/train_liar_raw.sh
  ├─ 11 inference             scripts/infer_<dataset>.py
  └─ 12 scoring               scripts/evaluate.py → scripts/report_results.py
```

These are the scripts named in the project runbooks, vendored unchanged:
`convert_pafc_to_training_text.py`, `add_labels_by_claim.py`,
`convert_to_compact_format.py`, `convert_to_llamafactory_robust.py`.

An alternative route exists for data already in chat/`messages` form —
`scripts/alternative/chat_build_training_text.py` then `chat_render_compact.py`, which reaches the
same place. The numbered route above is the one that produced the published
datasets; prefer it.

## How faithful is this?

Running the canonical route on the current AVeriTeC dev features and comparing
against the released `pafc_top10_eval.json`:

```
instruction identical : True
system identical      : True
output label identical: True
compact line format   : identical
full input byte-equal : 1 of 3 sampled claims
```

**Stage 7 is not optional.** Stage 6 emits verbose multi-line feature blocks
(`- trust_reasons: …`). The verifiers were fine-tuned on the *compact* one-line
form that stage 7 produces. Training on stage-6 output gives prompts that do not
match training and will not reproduce the reported numbers.

One command runs stages 1–8 for a split:

```bash
bash scripts/run_pipeline.sh --dataset averitec --split train \
     --retrieval data/raw/averitec/train_retrieval.json \
     --qa        data/raw/averitec/train.json \
     --out       work/averitec
```

---

The two that differ do so in the **answer text itself**, not in truncation or
formatting — the released file contains rewritten answers (`"… it was a spoof.
Context from the evidence: …"`) that do not appear anywhere in the current
feature file. HerO2's `answer_rewriting.py` is an LLM generation step
(Qwen3-8B) that was never seeded, so its output is not reproducible run to run.

**Consequence:** this pipeline reproduces the *format* exactly and the
*methodology* exactly, but it cannot reproduce the released SFT files
byte-for-byte, because an upstream non-deterministic stage sits between the
corpus and the features. Retrain from the released SFT files if you need the
exact published numbers; regenerate from scratch if you are running the method
on new data. This is also why 12 train + 2 eval records in the earlier audit
could not be regenerated.

## Stage 0 — retrieval (not included)

This repo starts from retrieved evidence. Upstream of stage 1, AVeriTeC runs
HerO2 unchanged — `hyde_fc_generation.py` → `retrieval.py` →
`question_generation.py` → `answer_rewriting.py` — with **one patch**: retrieval
emits richer rows so F2 has something to work with. The ranking is identical; only
the output record grows:

```json
{"rank": 1, "source_doc_idx": 27, "retrieval_score": 0.7812,
 "sentence": "...", "url": "https://web.archive.org/web/.../https://example.com/p",
 "canonical_source_url": "https://example.com/p",
 "archive_url": "https://web.archive.org/web/.../https://example.com/p",
 "is_archive_copy": true, "domain": "example.com"}
```

`canonical_source_url`, `is_archive_copy` and `domain` are what F2 matches
against the inventory, so a plain HerO2 retrieval file will not work. PA-FC
changes *which fields are recorded*, never which evidence is selected or in what
order. The other datasets use their own reference retrievers (TakeItEasy for
PubHealth, REFLEX for LIAR-RAW). Point `--retrieval` at your own
`<split>_retrieval_top_k_with_idx.json`; `docs/DATASETS.md` gives the shape.

## Stage 7 in detail — what "compact" means

Stages 1–4 leave four features hanging off each evidence item, and stage 6 writes
them out verbosely. Stage 7 collapses them into **one line of text** per evidence
item, and nothing else about the prompt changes:

```
[E1]
Q: Did the UK record fewer deaths in weeks 1-15 of 2020 than 2018?
A: ONS data show 2,760 fewer registered deaths ...

NLI: refute (0.94) | SRC: fact_checker | TRUST: high | PROV: weak | FLAGS: fact_check, leakage_sensitive
```

Two properties this repo enforces, because the whole matched design rests on them:

- **Only the metadata line varies.** Claim text, evidence text, evidence order
  and top-k are byte-identical across conditions. `src/pafc/compact.py` rewrites
  that line in place on an existing prompt file rather than regenerating prompts,
  so a drifting serializer cannot silently change the comparison.
- **`FLAGS` has a fixed order** — `leakage_sensitive, archive_copy, fact_check,
  primary_source, low_credibility, satire, reference_data, multimedia`. The order
  is load-bearing: it is part of the token sequence the model was trained on.
  `tests/test_compact_render.py` pins it.

Stage 7 also truncates each answer to its first 2 sentences (`--answer-sentences`),
which is what keeps top-10 prompts inside the 4096-token context. Measured
reduction on the released data is roughly 45–50% of tokens.

Stage 8 then wraps those records into the Alpaca array LLaMA-Factory expects —
`{instruction, input, output, system}` — and writes a `dataset_info` snippet to
merge into LLaMA-Factory's `data/dataset_info.json`. The `instruction` and
`system` strings it emits are byte-identical to those in the released training
file.

---

## Stage 7 — hyperparameters are per-dataset

This is the single easiest thing to get wrong. The four datasets do **not** share
a training recipe, and three of the four do not share a trainer either.

| Dataset | Trainer | batch | accum | lr | warmup | steps |
|---|---|---:|---:|---:|---:|---:|
| AVeriTeC | LLaMA-Factory | 2 | 8 | **1e-4** | 0.05 | 924 |
| RAGuard | LLaMA-Factory | 1 | 8 | **2e-4** | 0.03 | 795 |
| PubHealth | TakeItEasy-CausalLM (`takeiteasy/`) | 1 | 16 | **2e-4** | 0.03 | 1839 |
| LIAR-RAW | REFLEX (`scripts/train_reflex.py`) | 1 | 96 | **2e-5** | — | 195 |

Shared across all four: QLoRA 4-bit, LoRA rank 16, α 32, dropout 0.05, 3 epochs,
4096-token context, cosine schedule, bf16.

LIAR-RAW's 2e-5 is an order of magnitude below the others. Training it at 1e-4
does not reproduce the reported numbers.

```bash
# AVeriTeC / PubHealth / RAGuard
llamafactory-cli train configs/train/qwen/averitec_pafc.yaml
llamafactory-cli train configs/train/llama/averitec_pafc.yaml

# LIAR-RAW
bash scripts/train_liar_raw.sh --backbone qwen --seed 42 \
     --train work/liar_raw/sft/train.jsonl --out saves/pafc/liar_raw_qwen_seed42

# all three seeds for one cell
bash scripts/train_all_seeds.sh --config configs/train/qwen/averitec_pafc.yaml
```

### PubHealth has two trainers

PubHealth's published numbers come from TakeItEasy's native data pipeline with a
CausalLM head swapped in (native TakeItEasy is T5-only). Its seed-42 run used
LLaMA-Factory instead and is excluded.
The hyperparameters were matched deliberately, but they are still two
implementations. This repo reports PubHealth from the TakeItEasy-CausalLM seeds
only rather than averaging across the two, because averaging across trainers
would hide an implementation difference inside a seed mean. The README names
the seed behind every reproduced number.

---

## Stages 8–9 — inference and scoring

```bash
python scripts/infer_averitec.py --config configs/train/qwen/averitec_pafc.yaml \
       --test_file work/averitec/sft/eval.jsonl \
       --adapter_dir saves/pafc/averitec_qwen_seed42 \
       --out_dir preds/averitec_qwen_seed42

python scripts/evaluate.py --predictions preds/averitec_qwen_seed42/predictions.jsonl

cp configs/runs.example.json configs/runs.json   # point at your prediction dirs
python scripts/report_results.py --runs configs/runs.json --csv results.csv
```

`report_results.py` recomputes every number from per-claim predictions. No paper
value is hardcoded. A cell backed by fewer than three seeds is printed `partial`
rather than presented as a three-seed mean.
