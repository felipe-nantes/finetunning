"""Shared helpers for the QLoRA cybersecurity pipeline."""
from __future__ import annotations

import hashlib
import re
import yaml
from datasketch import MinHash, MinHashLSH

SYSTEM_PROMPT = (
    "You are a cybersecurity assistant for authorized penetration testing and "
    "defense. You explain methodology, tools, output interpretation, and "
    "mitigations. You help only with authorized, legal security work and you "
    "refuse to produce functional malware or weaponized exploits. Answer in the "
    "user's language."
)

_REFUSAL_PATTERNS = [
    r"\bi cannot help\b", r"\bi can't help\b", r"\bi'?m unable to (assist|help)\b",
    r"\bas an ai language model\b", r"\bi cannot (assist|provide|comply)\b",
]
_THEMES = {
    "recon": ["recon", "reconnaissance", "enumerat", "subdomain", "osint", "nmap", "port scan"],
    "web": ["xss", "sql injection", "sqli", "csrf", "idor", "web app", "burp", "cookie", "http"],
    "network": ["tcp", "udp", "packet", "firewall", "vlan", "arp", "dns tunnel", "smb", "wireshark"],
    "privesc": ["privilege escalation", "privesc", "suid", "sudo", "kernel exploit", "token impersonation"],
    "defense": ["blue team", "detect", "siem", "mitigation", "hardening", "incident", "monitor", "defen"],
    "cve": ["cve-", "cwe-", "vulnerability disclosure", "patch", "advisory"],
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def detect_lang(text: str) -> str:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    try:
        code = detect(text)
    except Exception:
        return "other"
    if code == "en":
        return "en"
    if code == "pt":
        return "pt"
    return "other"


def response_len_ok(text: str, lo: int = 80, hi: int = 1500, tokenizer=None) -> bool:
    if tokenizer is not None:
        n = len(tokenizer(text)["input_ids"])
    else:
        n = len(text.split())
    return lo <= n <= hi


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _REFUSAL_PATTERNS)


def tag_theme(text: str) -> str:
    low = text.lower()
    best, best_hits = "other", 0
    for theme, kws in _THEMES.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > best_hits:
            best, best_hits = theme, hits
    return best


def _row_text(row: dict) -> str:
    return " ".join(m["content"] for m in row["messages"])


def exact_dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for row in rows:
        h = hashlib.sha256(_row_text(row).encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(row)
    return out


def _minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for tok in set(text.lower().split()):
        m.update(tok.encode("utf-8"))
    return m


def minhash_dedup(rows: list[dict], threshold: float = 0.8, num_perm: int = 128) -> list[dict]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    out = []
    for i, row in enumerate(rows):
        m = _minhash(_row_text(row), num_perm)
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        out.append(row)
    return out


def to_prompt_completion(example: dict, tokenizer) -> dict:
    msgs = example["messages"]
    assert msgs[-1]["role"] == "assistant", "last message must be assistant"
    prompt = tokenizer.apply_chat_template(
        msgs[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    completion = msgs[-1]["content"] + tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}
