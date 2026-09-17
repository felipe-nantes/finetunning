"""Build the public data layer (Camada A) into canonical JSONL + manifest."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from scripts import common

SOURCES = [
    ("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", "trendyol", "apache-2.0"),
]

# Sources that were evaluated for the public layer and dropped. Kept here (and
# copied into the manifest) so the dataset card can say why they are absent
# instead of silently listing only what survived.
EXCLUDED_SOURCES = [
    {
        "repo": "AlicanKiraz0/Cybersecurity-Dataset-v1",
        "reason": (
            "100% das 2390 linhas normalizadas sao duplicata exata (SHA256 do "
            "texto user+assistant) de linhas do Trendyol/Trendyol-Cybersecurity-"
            "Instruction-Tuning-Dataset -- inclusive o campo system bruto de "
            "ambos os datasets originais no HuggingFace coincide. Nenhuma das "
            "duas dataset cards credita a outra. Verificado em 2026-09-16."
        ),
        "kept_pre_dedup": 2390,
        "kept_post_dedup": 0,
    },
]


def normalize_row(raw: dict, source: str) -> dict | None:
    user = (raw.get("user") or "").strip()
    assistant = (raw.get("assistant") or "").strip()
    system = common.SYSTEM_PROMPT
    if not user or not assistant:
        return None
    if common.detect_lang(user) != "en":
        return None
    if not common.response_len_ok(assistant):
        return None
    if common.is_refusal(assistant):
        return None
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "lang": "en",
        "theme": common.tag_theme(user + " " + assistant),
    }


def filter_by_token_budget(rows, tokenizer, max_length: int, min_completion_tokens: int = 64):
    """Keep only rows that survive the training tokenizer intact.

    The word-count prefilter in `normalize_row` is cheap but does not line up
    with the `max_length` truncation applied in train/eval: a row longer than
    that budget would reach the model with its answer chopped mid-sentence,
    teaching it to stop abruptly. Here the row is rendered exactly as training
    renders it and measured in tokens, so nothing that stays is ever truncated.

    Returns (kept_rows, stats).
    """
    kept = []
    stats = {"seen": len(rows), "over_max_length": 0, "under_min_completion": 0,
             "max_length": max_length, "min_completion_tokens": min_completion_tokens}
    for row in rows:
        if common.rendered_token_len(row, tokenizer) > max_length:
            stats["over_max_length"] += 1
            continue
        if common.completion_token_len(row, tokenizer) < min_completion_tokens:
            stats["under_min_completion"] += 1
            continue
        kept.append(row)
    stats["kept"] = len(kept)
    return kept, stats


def stratified_cap(rows: list[dict], cap: int, key: str = "theme", seed: int = 42) -> list[dict]:
    if len(rows) <= cap:
        return list(rows)
    rng = random.Random(seed)
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out, exhausted = [], set()
    # round-robin so small classes are fully kept and large ones are trimmed
    while len(out) < cap and len(exhausted) < len(buckets):
        for name, b in buckets.items():
            if name in exhausted:
                continue
            if not b:
                exhausted.add(name); continue
            out.append(b.pop())
            if len(out) >= cap:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=-1, help="debug: cap raw rows per source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/0.6b.yaml",
                    help="run config: supplies the tokenizer and the max_length the filter aligns to")
    ap.add_argument("--min-completion-tokens", type=int, default=64)
    ap.add_argument("--no-oos", action="store_true",
                    help="skip the curated out-of-scope refusal pairs")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    run_cfg = common.load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model_name"])
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"sources": [], "excluded_sources": EXCLUDED_SOURCES,
                "cap": args.cap, "seed": args.seed}
    all_rows = []
    for repo, source, lic in SOURCES:
        ds = load_dataset(repo, split="train")
        if args.limit > 0:
            ds = ds.select(range(min(args.limit, len(ds))))
        raw_n = len(ds)
        kept = [r for r in (normalize_row(x, source) for x in ds) if r]
        manifest["sources"].append(
            {"repo": repo, "source": source, "license": lic, "raw": raw_n, "kept": len(kept)}
        )
        all_rows.extend(kept)

    all_rows, len_stats = filter_by_token_budget(
        all_rows, tokenizer, run_cfg["max_length"], args.min_completion_tokens
    )
    len_stats["tokenizer"] = run_cfg["model_name"]
    manifest["length_filter"] = len_stats

    before = len(all_rows)
    all_rows = common.exact_dedup(all_rows)
    all_rows = common.minhash_dedup(all_rows, threshold=0.8)
    manifest["deduped_from"] = before
    manifest["after_dedup"] = len(all_rows)

    all_rows = stratified_cap(all_rows, args.cap, seed=args.seed)

    # Curated out-of-scope -> refusal pairs, added AFTER dedup/cap on purpose:
    # they share a refusal template, so MinHash would collapse them into a
    # couple of survivors. We want every scenario in, so they bypass dedup and
    # ride along into the shuffle/split like any other row.
    if not args.no_oos:
        from scripts import oos_pairs
        oos = oos_pairs.build_oos_rows()
        all_rows.extend(oos)
        manifest["oos_refusals"] = {"added": len(oos),
                                    "en": sum(r["lang"] == "en" for r in oos),
                                    "pt": sum(r["lang"] == "pt" for r in oos)}

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_frac))
    val, train = all_rows[:n_val], all_rows[n_val:]
    if len(train) == 0:
        raise SystemExit("not enough rows to split; increase --cap/--limit")
    manifest["train"] = len(train)
    manifest["val"] = len(val)
    manifest["theme_counts"] = _counts(all_rows)

    _write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _counts(rows):
    c = {}
    for r in rows:
        c[r["theme"]] = c.get(r["theme"], 0) + 1
    return c


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
