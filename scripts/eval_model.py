"""Evaluate base vs adapter: MCQ accuracy, val loss, qualitative + regression samples.
Run on the desktop:  uv run python scripts/04_eval.py --config configs/1.7b.yaml
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts import common

CYBERMETRIC_URL = "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json"


def parse_cybermetric(obj: dict) -> list[dict]:
    out = []
    for q in obj.get("questions", []):
        out.append({"question": q["question"], "choices": q["answers"], "answer": q["solution"]})
    return out


def score_choice(logprobs_by_letter: dict[str, float]) -> str:
    return max(logprobs_by_letter, key=logprobs_by_letter.get)


def bootstrap_ci(flags: list[int], n_boot: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    n = len(flags)
    acc = sum(flags) / n
    means = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return acc, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def completion_labels(prompt_ids: list[int], completion_ids: list[int]) -> list[int]:
    return [-100] * len(prompt_ids) + list(completion_ids)


def _load_model(model_name, adapter=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, dtype=torch.float32, device_map={"": 0},
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def _letter_token_id(tok, letter: str) -> int:
    ids = tok(" " + letter, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"letter {letter!r} is not a single token")
    return ids[0]


@torch.no_grad()
def _letter_logprob(tok, model, question, choices, letter):
    letter_id = _letter_token_id(tok, letter)
    prompt = f"{question}\n" + "\n".join(f"{k}. {v}" for k, v in choices.items()) + "\nAnswer:"
    ids = tok(prompt + " " + letter, return_tensors="pt").input_ids.to("cuda")
    if ids[0, -1].item() != letter_id:
        raise ValueError("in-context letter token mismatch")
    out = model(ids)
    logprobs = torch.log_softmax(out.logits[0, -2], dim=-1)  # predicts the letter token
    return logprobs[letter_id].item()


@torch.no_grad()
def mcq_accuracy(tok, model, items):
    flags = []
    for it in items:
        lp = {L: _letter_logprob(tok, model, it["question"], it["choices"], L) for L in it["choices"]}
        flags.append(int(score_choice(lp) == it["answer"]))
    return flags


def val_example(tok, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    full_ids = tok(prompt + completion, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        print(f"warning: prompt/completion tokenization seam mismatch for prompt={prompt!r}")
    labels = completion_labels(prompt_ids, full_ids[len(prompt_ids):])
    return full_ids, labels


@torch.no_grad()
def val_loss(tok, model, rows, max_examples: int = 100) -> float:
    losses = []
    for row in rows[:max_examples]:
        pc = common.to_prompt_completion(row, tok)
        full_ids, labels = val_example(tok, pc["prompt"], pc["completion"])
        input_ids = torch.tensor([full_ids]).to("cuda")
        label_ids = torch.tensor([labels]).to("cuda")
        out = model(input_ids=input_ids, labels=label_ids)
        losses.append(out.loss.item())
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


@torch.no_grad()
def _generate(tok, model, prompt, max_new_tokens=400):
    msgs = [{"role": "system", "content": common.SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit-mcq", type=int, default=500)
    ap.add_argument("--val-file", default="data/processed/val.jsonl")
    ap.add_argument("--max-val", type=int, default=100)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    adapter = os.path.join(cfg["output_dir"], "adapter")

    with urllib.request.urlopen(CYBERMETRIC_URL, timeout=60) as r:
        items = parse_cybermetric(json.load(r))[: args.limit_mcq]

    with open(args.val_file, encoding="utf-8") as fh:
        val_rows = [json.loads(l) for l in fh]

    results, samples = {}, []
    with open("eval/prompts.jsonl", encoding="utf-8") as fh:
        q_prompts = [{"set": "qual", **json.loads(l)} for l in fh]
    with open("eval/regression_prompts.jsonl", encoding="utf-8") as fh:
        r_prompts = [{"set": "regression", **json.loads(l)} for l in fh]

    for label, adapt in [("base", None), ("adapter", adapter)]:
        tok, model = _load_model(cfg["model_name"], adapt)
        flags = mcq_accuracy(tok, model, items)
        acc, lo, hi = bootstrap_ci(flags)
        results[label] = {"mcq_acc": acc, "ci95": [lo, hi], "n": len(flags),
                          "val_loss": val_loss(tok, model, val_rows, args.max_val)}
        for p in q_prompts + r_prompts:
            samples.append({"set": p["set"],
                            "lang": p["lang"], "prompt": p["prompt"],
                            "model": label, "output": _generate(tok, model, p["prompt"])})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open("docs/eval_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    _write_samples_md(samples)
    print(json.dumps(results, indent=2))


def _write_samples_md(samples):
    by_prompt = {}
    for s in samples:
        by_prompt.setdefault(s["prompt"], {})[s["model"]] = s["output"]
    lines = ["# Sample outputs: base vs adapter\n"]
    for prompt, outs in by_prompt.items():
        lines.append(f"## {prompt}\n")
        lines.append(f"**base:**\n\n{outs.get('base','').strip()}\n")
        lines.append(f"**adapter:**\n\n{outs.get('adapter','').strip()}\n\n---\n")
    with open("docs/samples.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
