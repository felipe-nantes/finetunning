"""Publish adapter, GGUF, and dataset to the Hugging Face Hub with generated cards.
Run after `huggingface-cli login` (or with HF_TOKEN set in the environment):
  uv run python scripts/06_push_hub.py --config configs/0.6b.yaml --hours <measured>
--user defaults to the account that owns the token; --gpu names the card that
actually trained the adapter, so the model card never claims the wrong hardware.
"""
from __future__ import annotations

import argparse
import json
import os

from scripts import common


def _eval_row(label: str, m: dict) -> str:
    """One row of the eval table, tolerant of a run that only has MCQ accuracy."""
    acc = f"{m['mcq_acc']:.3f}"
    ci = m.get("ci95")
    ci_txt = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
    vl = m.get("val_loss")
    vl_txt = f"{vl:.3f}" if isinstance(vl, (int, float)) else "—"
    return f"| {label} | {acc} | {ci_txt} | {vl_txt} |"


def _sources_block(manifest: dict) -> str:
    lines = [f"- `{s['repo']}` — {s['license']} — {s.get('kept', '?')} rows (public)"
             for s in manifest.get("sources", [])]
    syn = manifest.get("synthetic", {}).get("by_source", {})
    lines += [f"- {k} — generated locally via Ollama from licensed seeds — {v} rows"
              for k, v in syn.items()]
    oos = manifest.get("oos_refusals", {}).get("added")
    if oos:
        lines.append(f"- synthetic-oos — curated out-of-scope→refusal pairs — {oos} rows")
    return "\n".join(lines) if lines else "- see the dataset repo `NOTICE` for per-source attribution"


def render_model_card(meta: dict) -> str:
    e = meta["eval"]
    manifest = meta.get("manifest", {})
    samples = (meta.get("samples_md") or
               "See `docs/samples.md` in the GitHub repo for base-vs-adapter outputs.")
    return f"""---
license: {meta['base_license']}
base_model: {meta['model_name']}
library_name: peft
pipeline_tag: text-generation
tags: [cybersecurity, pentest, qlora, lora, qwen3]
language: [en, pt]
---

# decria-sec (LoRA adapter)

QLoRA fine-tune of `{meta['model_name']}` into a bilingual (EN strong, PT
functional) assistant for **authorized** penetration testing and defense:
methodology, tooling, output interpretation, and mitigation. Trained on a single
**{meta['gpu']}** (Pascal, `sm_61`, no bf16 — all compute in fp32) in
~{meta['hours']} h. This repo holds the **LoRA adapter**; apply it on top of the
base model, or use the GGUF build for Ollama/llama.cpp.

## Intended use
Authorized security testing, study, and CTF. Explaining attacker methodology so
it can be tested for and defended against, under a signed scope.

## Out-of-scope / unintended use
**Not** for unauthorized access, or for generating functional malware,
weaponized exploits, or ready-to-run payloads. Requests of that kind are refused
and redirected to the authorized/defensive angle — a behavior trained from
dedicated out-of-scope→refusal pairs (EN/PT) and checked in evaluation.

## Base model & license
- Base: `{meta['model_name']}` — Apache 2.0.
- Adapter released under the same terms; see the GitHub repo `NOTICE`.

## Training data
{_sources_block(manifest)}

Published dataset license is `cc-by-sa-4.0` (share-alike umbrella covering OWASP
WSTG). CyberMetric is used for **evaluation only** and is never redistributed.

## Evaluation (CyberMetric-500, zero-shot, fp32)

| model | MCQ accuracy | 95% CI | val loss |
|---|---|---|---|
{_eval_row("base", e['base'])}
{_eval_row("adapter", e['adapter'])}

MCQ CI is bootstrap (2000 resamples); val loss is on the held-out split
(completion-only). Honest expectation: SFT on ~4k examples teaches focus,
format, language, and refusal — not new knowledge — so the MCQ move is small;
the visible change is in open-ended answer quality and consistent refusals.

## Qualitative samples (base vs adapter)
{samples}

## Training configuration
NF4 4-bit (double quant) + LoRA (r={meta.get('lora_r', 16)}, alpha={meta.get('lora_alpha', 32)}),
fp32 compute (`fp16=False`, `bf16=False`), sequence length {meta.get('max_length', 768)},
effective batch {meta.get('effective_batch', 16)}, {meta.get('epochs', 2)} epochs,
completion-only loss. Hardware: {meta['gpu']}, ~{meta['hours']} h wall time. Full
reproducible pipeline (data build, synthetic generation, training, eval, export)
is on GitHub.

## Limitations
- Small base model on 3 GB of VRAM: expect gaps on niche or very recent CVEs and
  occasional hallucinated specifics — verify commands before running them.
- PT is functional, not native-fluent; EN is stronger.
- Refusal is trained but not guaranteed; do not rely on it as a safety control.
- Not audited for production or safety-critical use.
"""


def render_dataset_card(manifest: dict) -> str:
    src_lines = "\n".join(
        f"- `{s['repo']}` — {s['license']} — {s['kept']} rows" for s in manifest.get("sources", [])
    )
    syn = manifest.get("synthetic", {}).get("by_source", {})
    syn_lines = "\n".join(f"- {k}: {v}" for k, v in syn.items())
    return f"""---
license: cc-by-sa-4.0
language: [en, pt]
tags: [cybersecurity, instruction-tuning]
---

# decria-sec dataset

Mixed public + synthetic instruction data for an authorized-pentest assistant.
Train {manifest.get('train','?')} / val {manifest.get('val','?')}.

## Public sources
{src_lines}

## Synthetic (generated locally via Ollama from licensed seeds)
{syn_lines}

License is `cc-by-sa-4.0` as the share-alike umbrella; see `NOTICE` for per-source
attribution. CyberMetric is used for evaluation only and is not included here.
"""


def _samples_excerpt(path: str, max_prompts: int = 3, max_chars: int = 1800) -> str:
    """First few base-vs-adapter blocks from docs/samples.md, truncated for the card."""
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    blocks = text.split("\n## ")[1:]  # drop the file title
    picked = "\n## ".join(blocks[:max_prompts]).strip()
    if not picked:
        return ""
    excerpt = "## " + picked
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "\n\n… (truncated; full set in `docs/samples.md`)"
    return excerpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=None, help="HF username; default: owner of the logged-in token")
    ap.add_argument("--config", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--gpu", default="GTX 1060 3GB", help="GPU that ran the training, as written on the model card")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run = cfg["output_dir"]
    with open("docs/eval_results.json", encoding="utf-8") as fh:
        ev = json.load(fh)
    with open("data/manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)

    meta = {"model_name": cfg["model_name"], "base_license": "apache-2.0",
            "hours": args.hours, "gpu": args.gpu, "eval": ev,
            "manifest": manifest,
            "samples_md": _samples_excerpt("docs/samples.md"),
            "max_length": cfg["max_length"],
            "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"],
            "effective_batch": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
            "epochs": cfg["num_train_epochs"]}

    from huggingface_hub import HfApi
    api = HfApi()
    user = args.user or api.whoami()["name"]

    hub_name = cfg.get("hub_name", "decria-sec")
    lora_repo = f"{user}/{hub_name}-lora"
    gguf_repo = f"{user}/{hub_name}-GGUF"
    data_repo = f"{user}/decria-sec-dataset"

    api.create_repo(lora_repo, exist_ok=True)
    with open(os.path.join(run, "adapter", "README.md"), "w", encoding="utf-8") as fh:
        fh.write(render_model_card(meta))
    api.upload_folder(folder_path=os.path.join(run, "adapter"), repo_id=lora_repo)

    api.create_repo(gguf_repo, exist_ok=True)
    api.upload_folder(folder_path=os.path.join(run, "gguf"), repo_id=gguf_repo,
                      allow_patterns=["*.gguf", "Modelfile"], ignore_patterns=["*-f16.gguf"])

    api.create_repo(data_repo, repo_type="dataset", exist_ok=True)
    with open("data/processed/README.md", "w", encoding="utf-8") as fh:
        fh.write(render_dataset_card(manifest))
    api.upload_folder(folder_path="data/processed", repo_id=data_repo, repo_type="dataset",
                      allow_patterns=["*.jsonl", "README.md"])
    api.upload_file(path_or_fileobj="NOTICE", path_in_repo="NOTICE", repo_id=data_repo, repo_type="dataset")
    api.upload_file(path_or_fileobj="data/manifest.json", path_in_repo="manifest.json",
                    repo_id=data_repo, repo_type="dataset")
    print("pushed:", lora_repo, gguf_repo, data_repo)


if __name__ == "__main__":
    main()
