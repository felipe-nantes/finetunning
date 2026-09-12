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
    ("AlicanKiraz0/Cybersecurity-Dataset-v1", "alicankiraz0", "apache-2.0"),
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
    args = ap.parse_args()

    from datasets import load_dataset
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"sources": [], "cap": args.cap, "seed": args.seed}
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

    before = len(all_rows)
    all_rows = common.exact_dedup(all_rows)
    all_rows = common.minhash_dedup(all_rows, threshold=0.8)
    manifest["deduped_from"] = before
    manifest["after_dedup"] = len(all_rows)

    all_rows = stratified_cap(all_rows, args.cap, seed=args.seed)
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
