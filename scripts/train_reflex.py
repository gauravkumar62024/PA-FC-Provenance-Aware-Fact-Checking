import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from peft import get_peft_model_state_dict
from transformers import Trainer

from llmzoo.datasets.datasets import make_supervised_data_module
from llmzoo.models import build_model


os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct"
    )
    lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to training JSON."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to validation JSON."},
    )
    task_template: str = field(default="liar_raw_cev")
    wandb_run_name: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="paged_adamw_8bit")
    model_max_length: int = field(default=1024)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    if not data_args.train_data_path:
        raise ValueError("--train_data_path is required")

    if not data_args.eval_data_path:
        raise ValueError("--eval_data_path is required")

    model, tokenizer = build_model(
        model_args=model_args,
        training_args=training_args,
    )

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    output_path = pathlib.Path(training_args.output_dir)
    checkpoints = list(output_path.glob("checkpoint-*"))

    if checkpoints:
        print(f"Resuming from checkpoint in {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    if model_args.lora:
        # FIX: the legacy alpaca-lora state_dict monkey-patch double-filters the PEFT
        # state dict on modern peft/transformers and writes an EMPTY (40-byte) adapter.
        # Modern PeftModel.save_pretrained already extracts the LoRA weights correctly.
        adapter_dir = output_path / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(
            adapter_dir,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(adapter_dir)

        import os as _os
        _f = adapter_dir / "adapter_model.safetensors"
        _sz = _os.path.getsize(_f) if _f.exists() else 0
        if _sz < 1_000_000:
            raise RuntimeError(
                f"Adapter save FAILED: {_f} is only {_sz} bytes (expected >1MB). "
                "LoRA weights were not written."
            )
        print(f"LoRA adapter saved to: {adapter_dir}  ({_sz/1e6:.1f} MB)")
    else:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
