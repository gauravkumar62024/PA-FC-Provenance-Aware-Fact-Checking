# Training with LLaMA-Factory

Two of the four datasets fine-tune through
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): **AVeriTeC** and
**RAGuard**. PubHealth uses the TakeItEasy CausalLM pipeline
([`../takeiteasy/README.md`](../takeiteasy/README.md)) and LIAR-RAW uses the
REFLEX trainer, so neither is registered here.

This folder does not vendor LLaMA-Factory. Install it yourself, then point the
setup script at your checkout.

## Setup

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,bitsandbytes]" && cd ..

bash scripts/prepare_data.sh                      # if not already done
bash llamafactory/setup.sh /path/to/LLaMA-Factory
```

The script does three things:

1. Unpacks the five LLaMA-Factory datasets into `<LF>/data/`.
2. Merges `dataset_info.json` into the checkout's own file. It **merges**, never
   overwrites, and writes a `.bak` first. A LLaMA-Factory checkout ships many
   built-in entries and clobbering them would break every other recipe.
3. Copies the training configs into `<LF>/examples/train_lora/pafc/`.

## Train

```bash
cd /path/to/LLaMA-Factory

llamafactory-cli train examples/train_lora/pafc/qwen/averitec_pafc.yaml
llamafactory-cli train examples/train_lora/pafc/llama/averitec_pafc.yaml
```

Swap `averitec_pafc.yaml` for `raguard_pafc.yaml`.
Each config sets `seed: 42`; run again at 191 and 999 and average, or use
`scripts/train_all_seeds.sh` from the repo root.

LIAR-RAW:

```bash
bash scripts/train_liar_raw.sh --backbone qwen --seed 42 \
     --train data/liar_raw/train.json --out saves/pafc/liar_raw_qwen_seed42
```

## Registered datasets

All are Alpaca format with a `system` column.

| Registered name | File | Records |
|---|---|---:|
| `pafc_averitec_train` | `pafc_averitec_train.json` | 4,913 |
| `pafc_averitec_eval` | `pafc_averitec_eval.json` | 500 |
| `pafc_raguard_train` | `pafc_raguard_train.json` | 2,118 |
| `pafc_raguard_eval` | `pafc_raguard_eval.json` | 264 |
| `pafc_raguard_test` | `pafc_raguard_test.json` | 266 |

If you change a `dataset:` name in a config, change it in
`llamafactory/dataset_info.json` too. LLaMA-Factory fails with
`Cannot find dataset ...` when the two drift apart.

## Hyperparameters differ per dataset

| Dataset | batch | accum | lr | warmup | steps |
|---|---:|---:|---:|---:|---:|
| AVeriTeC | 2 | 8 | 1e-4 | 0.05 | 924 |
| RAGuard | 1 | 8 | 2e-4 | 0.03 | 795 |

Shared: QLoRA 4-bit, LoRA rank 16, alpha 32, dropout 0.05, 3 epochs, cutoff
4096, cosine, bf16. Do not copy one dataset's config onto another.

PubHealth is absent from this folder on purpose: its published numbers come from
the TakeItEasy CausalLM pipeline, and the LLaMA-Factory route scores materially
lower on Macro-F1.

## Citation

LLaMA-Factory is third-party software under Apache-2.0. If you train with it,
cite it:

```bibtex
@inproceedings{zheng2024llamafactory,
  title     = {LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author    = {Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and
               Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for
               Computational Linguistics (Volume 3: System Demonstrations)},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  year      = {2024},
  url       = {http://arxiv.org/abs/2403.13372}
}
```

Verified against LLaMA-Factory 0.9.6.dev0.
