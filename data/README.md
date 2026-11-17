# Released PA-FC datasets

The final, training-ready files produced by the pipeline in
[`../docs/PIPELINE.md`](../docs/PIPELINE.md). Each one already carries the
compact provenance line on every evidence item, so they can be registered with
LLaMA-Factory and trained without rerunning feature generation.

| Dataset | Split | Records | File |
|---|---|---:|---|
| AVeriTeC | train | 4,913 | `averitec/train.json.gz` |
| AVeriTeC | eval (dev) | 500 | `averitec/eval.json.gz` |
| PubHealth | train | 9,804 | `pubhealth/train.tsv.gz` |
| PubHealth | test | 1,231 | `pubhealth/test.tsv.gz` |
| PubHealth | val | 1,223 | `pubhealth/val.tsv.gz` |
| RAGuard | train | 2,118 | `raguard/train.json.gz` |
| RAGuard | eval | 264 | `raguard/eval.json.gz` |
| RAGuard | test | 266 | `raguard/test.json.gz` |
| LIAR-RAW | train | 6,168 | `liar_raw/train.json.gz` |
| LIAR-RAW | test | 1,251 | `liar_raw/test.json.gz` |

Gzipped to stay under the 100 MB GitHub per-file limit.

```bash
bash scripts/prepare_data.sh          # unpack all ten
gunzip -k data/averitec/train.json.gz # or just one
```

## Format

AVeriTeC, PubHealth and RAGuard use the Alpaca schema LLaMA-Factory expects:

```json
{"instruction": "Classify the veracity of the claim below ...",
 "input": "CLAIM: ...\n\n[E1] Q: ...\nA: ...\nNLI: refute (0.94) | SRC: fact_checker | TRUST: high | PROV: weak | FLAGS: fact_check, leakage_sensitive",
 "output": "REFUTED",
 "system": "You are a fact verification system ..."}
```

LIAR-RAW uses `{"id", "conversations"}` because it trains under REFLEX rather
than LLaMA-Factory.

PubHealth is a 12-column TSV, not JSON, because it trains under the TakeItEasy
CausalLM pipeline. Its PA-FC annotation is a claim-level aggregated summary in
`pafc_features_text` carrying F1, F3 and F4 but **not** F2, rather than a
per-evidence compact line. See [`../takeiteasy/README.md`](../takeiteasy/README.md).

Labels are `SUPPORTED`, `REFUTED`, `NOT ENOUGH EVIDENCE`, `CONFLICTING` for
AVeriTeC and PubHealth; `TRUE` / `FALSE` for RAGuard; and a 3-way collapse for
LIAR-RAW (see [`../docs/DATASETS.md`](../docs/DATASETS.md)).

## Sources, licenses and citation

These are **derived** files. The claims and evidence text come from four public
corpora, each under its own license, and each must be cited if you use the
corresponding file here. PA-FC adds only the F1 to F4 annotations; it does not
alter claim text, evidence text, evidence order or top-k.

| Corpus | Source | License |
|---|---|---|
| AVeriTeC | Schlichtkrull et al., NeurIPS 2023 Datasets and Benchmarks | CC BY-NC 4.0 |
| PubHealth | Kotonya and Toni, EMNLP 2020 | MIT |
| LIAR-RAW | Yang et al. (CofCED), AAAI 2022, built on LIAR | research use, see upstream |
| RAGuard | RAGuard benchmark release | see upstream release |

Check the upstream terms before redistributing these files further. AVeriTeC in
particular is **non-commercial**. If a license here conflicts with your intended
use, regenerate the file from your own copy of the corpus with
`scripts/run_pipeline.sh` instead of using the copy shipped here.

Retrieved evidence is archived web content and reflects what those corpora
captured. It is not vetted for accuracy, and low-credibility and satirical
sources are present on purpose: labelling them is what PA-FC does.
