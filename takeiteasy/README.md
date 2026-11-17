# PubHealth: TakeItEasy CausalLM pipeline

PubHealth does **not** train through LLaMA-Factory. It uses TakeItEasy's data
pipeline with a CausalLM verifier swapped in, because native TakeItEasy supports
only T5 (`AutoModelForSeq2SeqLM`).

This matters for reproduction. The paper's PubHealth row comes from this
pipeline. Training the same claims as per-evidence LLaMA-Factory prompts gives a
materially lower Macro-F1, so the two are not interchangeable.

## Representation

PubHealth conditions on a **claim-level aggregated summary**, one string per
claim, rather than a per-evidence compact line:

```
F1 relation counts: nei=10 ; F1 average confidence: 0.939 ;
F3 trust levels: medium=7, moderately_high=3 ; F3 average trust score: 0.479 ;
F4 provenance levels: weak=7, very_weak=2, moderate=1 ;
F4 flags: is_reference_or_data ; PA-FC evidence items summarized: 10
```

It carries **F1, F3 and F4 but not F2**. The data directory is named
`pubhealth_pafc_f134_fixed` for that reason. This is a deliberate difference
from AVeriTeC, RAGuard and LIAR-RAW, which all use the per-evidence form with
F2.

## Data

`../data/pubhealth_pafc_f134_fixed/{train,test,val}.tsv` after `scripts/prepare_data.sh`.
9,804 train / 1,231 test / 1,223 val. Twelve columns; the PA-FC annotation lives
in `pafc_features_text`, with the parsed form in `pafc_features_json` and a join
status in `pafc_match_status`.

## Train and infer

```bash
bash takeiteasy/run_pubhealth_causal_seed.sh
```

The runner trains Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct at seeds 191 and
999, then runs inference. Settings: lr 2e-4, 3 epochs, batch 1 with grad-accum
16, LoRA r16 alpha32, cutoff 4096, QLoRA 4-bit, 1,839 steps.

Seed 42 for PubHealth was trained under LLaMA-Factory rather than this pipeline,
so it is not part of the seed set here; mixing the two would hide an
implementation difference inside a seed mean.

To run one cell directly:

```bash
python3 takeiteasy/training_causal.py  --help
python3 takeiteasy/inference_causal.py --help
```

## Files

| File | Role |
|---|---|
| `causal_utils.py` | prompt construction and truncation |
| `training_causal.py` | QLoRA SFT loop |
| `inference_causal.py` | batched inference |
| `load_custom_dataset.py` | TSV loading |
| `feature_conversion_methods.py` | prompt formatting and label maps |
| `run_pubhealth_causal_seed.sh` | seed runner |

`causal_utils.py` truncates the **user body**, never the chat scaffolding. The
naive approach silently masked every label (`train_loss = 0`) whenever a prompt
exceeded the context window, so training aborts if total supervised tokens is 0.

These files are adapted from the TakeItEasy codebase of Yang and Rocha, "Take It
Easy: Label-Adaptive Self-Rationalization for Face Forgery Detection and
Localization" (2024), reference [36] in the paper. Check the upstream license
before reusing them.
