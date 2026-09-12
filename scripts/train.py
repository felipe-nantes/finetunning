"""QLoRA training. Run on the desktop:
  uv run python scripts/03_train.py --config configs/1.7b.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        per_device_eval_batch_size=1,
        report_to=["tensorboard"],
        seed=cfg["seed"],
    )


def force_fp32_trainable(model) -> int:
    """Cast every trainable parameter back to fp32 (TRL casts them to bf16 for 4-bit models).

    TRL 1.13's SFTTrainer.__init__ unconditionally casts trainable params of a
    quantized model to bf16 (see trl/trainer/sft_trainer.py), regardless of the
    fp16/bf16 flags in SFTConfig. The GTX 1060 (Pascal, sm_61) has no native
    bf16 support, so this must be undone right after the trainer is built.

    Returns the number of parameters cast.
    """
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
            n += 1
    return n


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
    n_cast = force_fp32_trainable(trainer.model)
    bad = [name for name, p in trainer.model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    assert not bad, f"non-fp32 trainable params: {bad[:5]}"
    print(f"trainable params cast back to fp32: {n_cast}")
    ckpt = _last_checkpoint(cfg["output_dir"])
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(os.path.join(cfg["output_dir"], "adapter"))
    tok.save_pretrained(os.path.join(cfg["output_dir"], "adapter"))
    print("saved adapter to", os.path.join(cfg["output_dir"], "adapter"))


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _last_checkpoint(output_dir: str):
    """Return the newest checkpoint-N dir that actually finished saving.

    A power cut mid-save can leave a partial checkpoint-N directory (no
    trainer_state.json). Resuming from that directory would fail, so we walk
    candidates newest-first and return the first one that looks complete.
    """
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        m = _CHECKPOINT_RE.match(name)
        if m and os.path.isdir(os.path.join(output_dir, name)):
            candidates.append((int(m.group(1)), os.path.join(output_dir, name)))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for _, path in candidates:
        if os.path.isfile(os.path.join(path, "trainer_state.json")):
            return path
    return None


if __name__ == "__main__":
    main()
