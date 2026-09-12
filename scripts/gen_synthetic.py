"""Synthetic data layer (Camada B): generate Q/A pairs from licensed seeds via Ollama."""
from __future__ import annotations

import argparse
import json
import os
import re
import random
import urllib.request

from scripts import common

OLLAMA_URL = "http://localhost:11434/api/chat"
STOPWORDS = set("the a an of to and or for with in on is are be how what "
                "why when your you i it this that as by from at".split())

ATTACK_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
NMAP_URL = "https://raw.githubusercontent.com/nmap/nmap/master/docs/nmap.usage.txt"
CWE_VIEW_URL = "https://cwe-api.mitre.org/api/v1/cwe/view/1430"          # CWE Top 25 (2024) view
CWE_WEAKNESS_URL = "https://cwe-api.mitre.org/api/v1/cwe/weakness/{ids}"  # comma-separated ids
WSTG_REPO = "https://github.com/OWASP/wstg"


def parse_cwe_members(view_json: dict) -> list[str]:
    """Return the CWE ids listed in a CWE view response."""
    views = view_json.get("Views") or []
    if not views:
        return []
    return [m["CweID"] for m in views[0].get("Members", []) if m.get("CweID")]


def parse_cwe_weaknesses(weak_json: dict) -> list[dict]:
    """Return [{'id','name','description'}] from a CWE weakness response."""
    out = []
    for w in weak_json.get("Weaknesses", []):
        if w.get("Name") and w.get("Description"):
            out.append({"id": str(w.get("ID", "")), "name": w["Name"], "description": w["Description"]})
    return out


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "finetunning-decria/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_to(url: str, dest: str, timeout: int = 120) -> None:
    """Fetch url fully into memory first; only then (over)write dest.

    Never opens dest before the fetch succeeds, so a failed/interrupted
    download cannot leave a truncated 0-byte file that a later
    os.path.exists idempotency check would mistake for "already done".
    """
    data = _http_get(url, timeout=timeout)
    with open(dest, "wb") as fh:
        fh.write(data)


def fetch_seeds(seeds_dir: str) -> None:
    """Download every licensed seed source into seeds_dir (idempotent)."""
    import shutil
    import subprocess
    os.makedirs(os.path.join(seeds_dir, "tools"), exist_ok=True)
    os.makedirs(os.path.join(seeds_dir, "wstg"), exist_ok=True)

    attack = os.path.join(seeds_dir, "enterprise-attack.json")
    if not os.path.exists(attack):
        _download_to(ATTACK_URL, attack, timeout=600)

    nmap = os.path.join(seeds_dir, "tools", "nmap.txt")
    if not os.path.exists(nmap):
        _download_to(NMAP_URL, nmap)

    cwe = os.path.join(seeds_dir, "cwe_top25.json")
    if not os.path.exists(cwe):
        ids = parse_cwe_members(json.loads(_http_get(CWE_VIEW_URL)))
        weaknesses = parse_cwe_weaknesses(json.loads(_http_get(CWE_WEAKNESS_URL.format(ids=",".join(ids)))))
        with open(cwe, "w", encoding="utf-8") as fh:
            json.dump(weaknesses, fh, indent=2, ensure_ascii=False)

    wstg_src = os.path.join(seeds_dir, "wstg_src")
    if not os.path.isdir(os.path.join(wstg_src, "document")):
        # A previous interrupted clone may have left a partial wstg_src (or a
        # stale .partial dir); never treat either as done, and never clone
        # directly into the final path so a failure can't leave it half-done.
        partial = wstg_src + ".partial"
        if os.path.isdir(partial):
            shutil.rmtree(partial)
        subprocess.run(["git", "clone", "--depth", "1", WSTG_REPO, partial], check=True)
        if os.path.isdir(wstg_src):
            shutil.rmtree(wstg_src)
        os.rename(partial, wstg_src)
    src_docs = os.path.join(wstg_src, "document", "4-Web_Application_Security_Testing")
    for root, _, files in os.walk(src_docs):
        for f in files:
            if f.endswith(".md"):
                shutil.copy2(os.path.join(root, f), os.path.join(seeds_dir, "wstg", f))


def chunk_text(text: str, target_words: int = 400) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + target_words]) for i in range(0, len(words), target_words)]


def _key_terms(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
    return {t for t in toks if t not in STOPWORDS}


def anchored(answer: str, chunk: str, k: int = 2) -> bool:
    return len(_key_terms(answer) & _key_terms(chunk)) >= k


def _looks_like_pairs(candidate) -> bool:
    """True if candidate is a non-empty list of dicts, at least one of which
    carries both a "question" and an "answer" key. Filters out lists the
    model emitted for unrelated reasons ([], [1, 2, 3], ["a", "b"], ...)
    so they don't get mistaken for the actual Q/A array."""
    if not isinstance(candidate, list) or not candidate:
        return False
    if not all(isinstance(d, dict) for d in candidate):
        return False
    return any("question" in d and "answer" in d for d in candidate)


def parse_pairs(raw: str) -> list[dict]:
    data = None
    try:
        candidate = json.loads(raw)
        if _looks_like_pairs(candidate):
            data = candidate
    except json.JSONDecodeError:
        pass
    if data is None:
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch != "[":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw, i)
            except json.JSONDecodeError:
                continue
            if _looks_like_pairs(candidate):
                data = candidate
                break
    if data is None:
        return []
    out = []
    for d in data:
        if not isinstance(d, dict):
            continue
        q, a = (d.get("question") or "").strip(), (d.get("answer") or "").strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out


def build_gen_prompt(chunk: str, lang: str) -> str:
    if lang == "pt":
        return (
            "Você é um instrutor de segurança ofensiva autorizada. Com base APENAS no "
            "texto abaixo, gere de 2 a 3 pares pergunta/resposta técnicos em PORTUGUÊS, "
            "focados em teste autorizado, metodologia e mitigação. Não invente fatos fora "
            "do texto. Responda SÓ com um array JSON de objetos {\"question\",\"answer\"}.\n\n"
            f"TEXTO:\n{chunk}"
        )
    return (
        "You are an authorized offensive-security instructor. Using ONLY the text below, "
        "write 2-3 technical question/answer pairs in ENGLISH about authorized testing, "
        "methodology, and mitigation. Do not invent facts beyond the text. Reply with ONLY "
        "a JSON array of objects {\"question\",\"answer\"}.\n\n"
        f"TEXT:\n{chunk}"
    )


def ollama_chat(model: str, prompt: str, timeout: int = 180) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.7},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"]


def _load_seed_texts(seeds_dir: str) -> list[tuple[str, str]]:
    """Return (source_tag, text) units from downloaded seeds. See README for the fetch step."""
    units = []
    attack = os.path.join(seeds_dir, "enterprise-attack.json")
    if os.path.exists(attack):
        with open(attack, encoding="utf-8") as fh:
            bundle = json.load(fh)
        for obj in bundle.get("objects", []):
            if obj.get("type") == "attack-pattern" and obj.get("description"):
                units.append(("synthetic-attack", f"{obj.get('name','')}. {obj['description']}"))
    wstg = os.path.join(seeds_dir, "wstg")
    if os.path.isdir(wstg):
        for root, _, files in os.walk(wstg):
            for f in files:
                if f.endswith(".md"):
                    with open(os.path.join(root, f), encoding="utf-8", errors="ignore") as fh:
                        units.append(("synthetic-wstg", fh.read()))
    cwe = os.path.join(seeds_dir, "cwe_top25.json")
    if os.path.exists(cwe):
        with open(cwe, encoding="utf-8") as fh:
            for w in json.load(fh):
                units.append(("synthetic-cwe", f"{w['name']}. {w['description']}"))
    tools = os.path.join(seeds_dir, "tools")
    if os.path.isdir(tools):
        for f in os.listdir(tools):
            with open(os.path.join(tools, f), encoding="utf-8", errors="ignore") as fh:
                units.append(("synthetic-tool", fh.read()))
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", default="data/seeds")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--pt-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fetch-seeds", action="store_true")
    args = ap.parse_args()

    if args.fetch_seeds:
        fetch_seeds(args.seeds_dir)
        print(f"seeds ready in {args.seeds_dir}")
        return

    rng = random.Random(args.seed)
    units = _load_seed_texts(args.seeds_dir)
    if not units:
        raise SystemExit(f"no seeds found in {args.seeds_dir}; run the fetch step (README).")
    rng.shuffle(units)

    rows, made = [], 0
    for source_tag, text in units:
        if made >= args.target:
            break
        for chunk in chunk_text(text, 400):
            if made >= args.target:
                break
            lang = "pt" if rng.random() < args.pt_frac else "en"
            try:
                raw = ollama_chat(args.model, build_gen_prompt(chunk, lang))
            except Exception as e:
                print(f"gen error ({source_tag}): {e}"); continue
            try:
                for pair in parse_pairs(raw):
                    if not common.response_len_ok(pair["answer"]):
                        continue
                    if not anchored(pair["answer"], chunk, k=2):
                        continue
                    rows.append({
                        "messages": [
                            {"role": "system", "content": common.SYSTEM_PROMPT},
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]},
                        ],
                        "source": source_tag, "lang": lang,
                        "theme": common.tag_theme(pair["question"] + " " + pair["answer"]),
                    })
                    made += 1
            except Exception as e:
                print(f"parse error ({source_tag}): {e}")
                continue
        print(f"progress: {made}/{args.target}")

    rows = common.exact_dedup(rows)
    rows = common.minhash_dedup(rows, threshold=0.8)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    _append_jsonl(os.path.join(args.out_dir, "val.jsonl"), rows[:n_val])
    _append_jsonl(os.path.join(args.out_dir, "train.jsonl"), rows[n_val:])
    _update_manifest(rows, args.model)
    print(f"synthetic added: {len(rows)} (val {n_val})")


def _append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _update_manifest(rows, model, path="data/manifest.json"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            man = json.load(fh)
    else:
        man = {}

    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1

    prev = man.get("synthetic")
    if prev:
        added = prev.get("added", 0) + len(rows)
        merged_by_source = dict(prev.get("by_source", {}))
        for k, v in by_source.items():
            merged_by_source[k] = merged_by_source.get(k, 0) + v
        models = list(prev.get("models", []))
        prev_model = prev.get("model")
        if prev_model and prev_model not in models:
            models.append(prev_model)
        if model not in models:
            models.append(model)
        man["synthetic"] = {"model": model, "models": models, "added": added, "by_source": merged_by_source}
    else:
        man["synthetic"] = {"model": model, "models": [model], "added": len(rows), "by_source": by_source}

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
