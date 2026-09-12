"""QLoRA training. Run on the desktop:
  uv run python scripts/03_train.py --config configs/1.7b.yaml
"""
from __future__ import annotations

import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig

from scripts import common

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )


def render_dataset(ds, tokenizer):
    cols = ds.column_names
    return ds.map(lambda ex: common.to_prompt_completion(ex, tokenizer), remove_columns=cols)


def build_sft_config(cfg: dict) -> SFTConfig:
    return SFTConfig(
        output_dir=cfg["output_dir"],
        max_length=cfg["max_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=cfg.get("max_steps", -1),
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        weight_decay=0.0,
        optim="adamw_torch",
        fp16=False, bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        report_to=["tensorboard"],
        seed=cfg["seed"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--val-file", default="data/processed/val.jsonl")
    args = ap.parse_args()
    cfg = common.load_config(args.config)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], quantization_config=bnb, dtype=torch.float32, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")
    if cfg.get("max_train_samples", -1) > 0:
        train_ds = train_ds.select(range(min(cfg["max_train_samples"], len(train_ds))))
    train_ds = render_dataset(train_ds, tok)
    val_ds = render_dataset(val_ds, tok)

    sft = build_sft_config(cfg)
    trainer = SFTTrainer(
        model=model, args=sft, train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, peft_config=build_lora_config(cfg),
    )
    ckpt = _last_checkpoint(cfg["output_dir"])
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(os.path.join(cfg["output_dir"], "adapter"))
    tok.save_pretrained(os.path.join(cfg["output_dir"], "adapter"))
    print("saved adapter to", os.path.join(cfg["output_dir"], "adapter"))


def _last_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    return get_last_checkpoint(output_dir)


if __name__ == "__main__":
    main()
