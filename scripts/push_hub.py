"""Publish adapter, GGUF, and dataset to the Hugging Face Hub with generated cards.
Run on the laptop after `huggingface-cli login`:
  uv run python scripts/06_push_hub.py --user <HF_USER> --config configs/1.7b.yaml
"""
from __future__ import annotations

import argparse
import json
import os

from scripts import common


def render_model_card(meta: dict) -> str:
    e = meta["eval"]
    return f"""---
license: {meta['base_license']}
base_model: {meta['model_name']}
tags: [cybersecurity, pentest, qlora, lora, qwen3]
language: [en, pt]
---

# decria-sec (LoRA adapter)

QLoRA fine-tune of `{meta['model_name']}` into an assistant for **authorized**
penetration testing and defense (methodology, tooling, output interpretation,
mitigation). Trained on a single **{meta['gpu']}** in ~{meta['hours']} h.

## Evaluation (CyberMetric-500, zero-shot)

| model | MCQ accuracy |
|---|---|
| base | {e['base']['mcq_acc']:.3f} |
| adapter | {e['adapter']['mcq_acc']:.3f} |

## Intended use
Authorized security testing, study, CTF. **Not** for unauthorized access or
generating functional malware. Out-of-scope requests are refused.

## Training
NF4 4-bit + LoRA (r={meta.get('lora_r', 16)}), fp32 compute, seq {meta.get('max_length', 768)},
effective batch {meta.get('effective_batch', 16)}, {meta.get('epochs', 2)} epochs. See the GitHub repo for the full reproducible pipeline.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--hours", type=float, required=True)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run = cfg["output_dir"]
    with open("docs/eval_results.json", encoding="utf-8") as fh:
        ev = json.load(fh)
    with open("data/manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)

    meta = {"model_name": cfg["model_name"], "base_license": "apache-2.0",
            "hours": args.hours, "gpu": "GTX 1060 6GB", "eval": ev,
            "max_length": cfg["max_length"],
            "lora_r": cfg["lora_r"],
            "effective_batch": cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"],
            "epochs": cfg["num_train_epochs"]}

    from huggingface_hub import HfApi
    api = HfApi()

    lora_repo = f"{args.user}/decria-sec-1.7b-lora"
    gguf_repo = f"{args.user}/decria-sec-1.7b-GGUF"
    data_repo = f"{args.user}/decria-sec-dataset"

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
