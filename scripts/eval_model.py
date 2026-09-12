"""Evaluate base vs adapter: MCQ accuracy, val loss, qualitative + regression samples.
Run on the desktop:  uv run python scripts/04_eval.py --config configs/1.7b.yaml
"""
from __future__ import annotations

import argparse
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


@torch.no_grad()
def _letter_logprob(tok, model, question, choices, letter):
    prompt = f"{question}\n" + "\n".join(f"{k}. {v}" for k, v in choices.items()) + "\nAnswer:"
    ids = tok(prompt + " " + letter, return_tensors="pt").input_ids.to("cuda")
    out = model(ids)
    logprobs = torch.log_softmax(out.logits[0, -2], dim=-1)  # predicts the letter token
    letter_id = tok(" " + letter, add_special_tokens=False).input_ids[-1]
    return logprobs[letter_id].item()


@torch.no_grad()
def mcq_accuracy(tok, model, items):
    flags = []
    for it in items:
        lp = {L: _letter_logprob(tok, model, it["question"], it["choices"], L) for L in it["choices"]}
        flags.append(int(score_choice(lp) == it["answer"]))
    return flags


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
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    adapter = os.path.join(cfg["output_dir"], "adapter")

    with urllib.request.urlopen(CYBERMETRIC_URL, timeout=60) as r:
        items = parse_cybermetric(json.load(r))[: args.limit_mcq]

    results, samples = {}, []
    q_prompts = [json.loads(l) for l in open("eval/prompts.jsonl", encoding="utf-8")]
    r_prompts = [json.loads(l) for l in open("eval/regression_prompts.jsonl", encoding="utf-8")]

    for label, adapt in [("base", None), ("adapter", adapter)]:
        tok, model = _load_model(cfg["model_name"], adapt)
        flags = mcq_accuracy(tok, model, items)
        acc, lo, hi = bootstrap_ci(flags)
        results[label] = {"mcq_acc": acc, "ci95": [lo, hi], "n": len(flags)}
        for p in q_prompts + r_prompts:
            samples.append({"set": "qual" if p in q_prompts else "regression",
                            "lang": p["lang"], "prompt": p["prompt"],
                            "model": label, "output": _generate(tok, model, p["prompt"])})
        del model
        torch.cuda.empty_cache()

    json.dump(results, open("docs/eval_results.json", "w", encoding="utf-8"), indent=2)
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
    open("docs/samples.md", "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    main()
