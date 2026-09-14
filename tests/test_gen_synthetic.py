import json
import os
import sys

import pytest

from scripts import gen_synthetic as gs


def test_chunk_text_word_bounds():
    text = "word " * 1000
    chunks = gs.chunk_text(text, target_words=200)
    assert all(len(c.split()) <= 220 for c in chunks)
    assert sum(len(c.split()) for c in chunks) >= 900


def test_parse_pairs_extracts_json_array():
    raw = 'noise\n[{"question":"What is XSS?","answer":"Cross site scripting ..."}] trailing'
    pairs = gs.parse_pairs(raw)
    assert pairs and pairs[0]["question"].startswith("What is XSS")


def test_parse_pairs_bad_json_returns_empty():
    assert gs.parse_pairs("not json at all") == []


def test_anchored_requires_shared_terms():
    chunk = "Lateral movement using SMB and PsExec across Windows hosts"
    assert gs.anchored("Explain SMB lateral movement with PsExec on Windows", chunk, k=2)
    assert not gs.anchored("How do I bake sourdough bread", chunk, k=2)


def test_build_gen_prompt_lang():
    pt = gs.build_gen_prompt("chunk text", lang="pt")
    assert "português" in pt.lower()
    assert "120" in pt and "300" in pt
    en = gs.build_gen_prompt("chunk text", lang="en")
    assert "json" in en.lower()
    assert "120" in en and "300" in en


def test_parse_cwe_members_and_weaknesses():
    view = {"Views": [{"ID": "1430", "Members": [{"CweID": "79", "ViewID": "1430"}, {"CweID": "89", "ViewID": "1430"}]}]}
    assert gs.parse_cwe_members(view) == ["79", "89"]
    weak = {"Weaknesses": [{"ID": "79", "Name": "XSS", "Description": "Improper neutralization"},
                           {"ID": "0", "Name": "", "Description": "skipped"}]}
    items = gs.parse_cwe_weaknesses(weak)
    assert items == [{"id": "79", "name": "XSS", "description": "Improper neutralization"}]


def test_parse_cwe_members_empty():
    assert gs.parse_cwe_members({}) == []


def test_parse_pairs_ignores_stray_brackets():
    raw = 'Sure, here are the pairs:\n[{"question":"What is XSS?","answer":"Cross site scripting"}]\nLet me know if you want more examples [like CSRF or SQLi].'
    pairs = gs.parse_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is XSS?"


def test_download_to_writes_only_on_success(tmp_path, monkeypatch):
    dest_ok = tmp_path / "ok.bin"
    monkeypatch.setattr(gs, "_http_get", lambda url, timeout=120: b"ok")
    gs._download_to("http://example.invalid/ok", str(dest_ok))
    assert dest_ok.read_bytes() == b"ok"

    dest_fail = tmp_path / "fail.bin"

    def _boom(url, timeout=120):
        raise OSError("network down")

    monkeypatch.setattr(gs, "_http_get", _boom)
    with pytest.raises(OSError):
        gs._download_to("http://example.invalid/fail", str(dest_fail))
    assert not dest_fail.exists()


def test_parse_pairs_skips_non_dict_lists():
    cases = [
        'Here are 3 examples: [1, 2, 3]\n[{"question": "What is XSS?", "answer": "Cross-site scripting"}]',
        'Options: ["a", "b"]\n[{"question": "What is XSS?", "answer": "Cross-site scripting"}]',
        'Nothing yet: []\n[{"question": "What is XSS?", "answer": "Cross-site scripting"}]',
    ]
    for raw in cases:
        pairs = gs.parse_pairs(raw)
        assert len(pairs) == 1
        assert pairs[0]["question"] == "What is XSS?"


def test_parse_pairs_empty_list_then_real():
    raw = '[]\n[{"question":"Q","answer":"A"}]'
    pairs = gs.parse_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "Q"


def test_update_manifest_accumulates(tmp_path):
    path = str(tmp_path / "manifest.json")
    gs._update_manifest(
        [{"source": "synthetic-attack"}, {"source": "synthetic-attack"}], "model-a", path=path, n_val=1
    )
    gs._update_manifest([{"source": "synthetic-cwe"}], "model-b", path=path, n_val=0)

    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    assert man["synthetic"]["added"] == 3
    assert man["synthetic"]["by_source"] == {"synthetic-attack": 2, "synthetic-cwe": 1}
    assert man["train"] == 2
    assert man["val"] == 1


def test_resolve_ollama_url_precedence_and_normalization():
    assert gs.resolve_ollama_url(None, None) == "http://localhost:11434/api/chat"
    assert gs.resolve_ollama_url(None, "192.168.0.10:11434") == "http://192.168.0.10:11434/api/chat"
    assert gs.resolve_ollama_url("http://host:1/", "x") == "http://host:1/api/chat"


def test_fetch_seeds_skips_readme_md(tmp_path):
    seeds_dir = tmp_path / "seeds"
    (seeds_dir / "tools").mkdir(parents=True)
    (seeds_dir / "wstg").mkdir(parents=True)
    (seeds_dir / "enterprise-attack.json").write_text("{}", encoding="utf-8")
    (seeds_dir / "tools" / "nmap.txt").write_text("usage", encoding="utf-8")
    (seeds_dir / "cwe_top25.json").write_text("[]", encoding="utf-8")

    # Pre-populate wstg_src/document so fetch_seeds treats the clone as already done.
    src_docs = seeds_dir / "wstg_src" / "document" / "4-Web_Application_Security_Testing"
    src_docs.mkdir(parents=True)
    (src_docs / "README.md").write_text("not a test case", encoding="utf-8")
    (src_docs / "4.1-Some_Test.md").write_text("real WSTG content", encoding="utf-8")

    gs.fetch_seeds(str(seeds_dir))

    copied = sorted(os.listdir(seeds_dir / "wstg"))
    assert copied == ["4.1-Some_Test.md"]


def test_main_raises_after_max_consecutive_ollama_failures(monkeypatch, tmp_path):
    fake_units = [("src1", "word " * 50), ("src2", "word " * 50), ("src3", "word " * 50)]
    monkeypatch.setattr(gs, "_load_seed_texts", lambda seeds_dir: fake_units)

    def _always_fails(model, prompt, url, timeout=180):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(gs, "ollama_chat", _always_fails)
    monkeypatch.setattr(sys, "argv", [
        "gen_synthetic.py",
        "--out-dir", str(tmp_path),
        "--max-consecutive-failures", "2",
    ])

    with pytest.raises(SystemExit, match="consecutive Ollama failures"):
        gs.main()


def test_chunk_key_is_stable_and_source_scoped():
    assert gs.chunk_key("a", "text") == gs.chunk_key("a", "text")
    assert gs.chunk_key("a", "text") != gs.chunk_key("b", "text")


def test_pick_lang_is_deterministic_and_respects_fraction():
    keys = [gs.chunk_key("s", f"chunk {i}") for i in range(400)]
    langs = [gs.pick_lang(k, 0.30) for k in keys]
    assert langs == [gs.pick_lang(k, 0.30) for k in keys]  # stable across resumes
    pt = langs.count("pt") / len(langs)
    assert 0.2 < pt < 0.4
    assert all(gs.pick_lang(k, 0.0) == "en" for k in keys)
    assert all(gs.pick_lang(k, 1.0) == "pt" for k in keys)


def test_load_raw_rows_drops_torn_last_line(tmp_path):
    p = tmp_path / "synthetic_raw.jsonl"
    p.write_text('{"messages": [], "source": "a"}\n{"messages": [], "sou', encoding="utf-8")
    rows = gs.load_raw_rows(str(p))
    assert len(rows) == 1
    assert gs.load_raw_rows(str(tmp_path / "missing.jsonl")) == []


def test_main_resumes_and_skips_done_chunks(monkeypatch, tmp_path):
    fake_units = [("src1", "alpha beta gamma delta"), ("src2", "epsilon zeta eta theta")]
    monkeypatch.setattr(gs, "_load_seed_texts", lambda seeds_dir: fake_units)
    calls = []

    def _fake_chat(model, prompt, url, timeout=180):
        calls.append(prompt)
        # answer echoes chunk terms so `anchored` passes, and is long enough
        terms = prompt.split("TEXT:\n")[-1].split("TEXTO:\n")[-1]
        return json.dumps([{"question": "Q?", "answer": (terms + " ") * 40}])

    monkeypatch.setattr(gs, "ollama_chat", _fake_chat)
    monkeypatch.setattr(gs, "_update_manifest", lambda *a, **k: None)

    argv = ["gen_synthetic.py", "--out-dir", str(tmp_path), "--target", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    gs.main()
    first_calls = len(calls)
    assert first_calls == 1
    assert (tmp_path / "train.jsonl").exists() or (tmp_path / "val.jsonl").exists()
    # the session files are cleared once merged
    assert not (tmp_path / "synthetic_raw.jsonl").exists()
    assert not (tmp_path / "synthetic_done.txt").exists()


def test_main_interrupted_run_keeps_rows_and_resumes(monkeypatch, tmp_path):
    fake_units = [("src1", " ".join(f"w{i}" for i in range(2000)))]
    monkeypatch.setattr(gs, "_load_seed_texts", lambda seeds_dir: fake_units)
    monkeypatch.setattr(gs, "_update_manifest", lambda *a, **k: None)
    seen = []

    def _chat_then_die(model, prompt, url, timeout=180):
        terms = prompt.split("TEXT:\n")[-1].split("TEXTO:\n")[-1]
        seen.append(terms[:20])
        if len(seen) == 3:
            raise KeyboardInterrupt("power cut")
        return json.dumps([{"question": "Q?", "answer": (terms[:200] + " ") * 10}])

    monkeypatch.setattr(gs, "ollama_chat", _chat_then_die)
    monkeypatch.setattr(sys, "argv", ["gen_synthetic.py", "--out-dir", str(tmp_path), "--target", "99"])
    with pytest.raises(KeyboardInterrupt):
        gs.main()

    raw = tmp_path / "synthetic_raw.jsonl"
    done = tmp_path / "synthetic_done.txt"
    assert len(gs.load_raw_rows(str(raw))) >= 2      # flushed before the crash
    assert len(gs.load_done_keys(str(done))) == 2    # the crashed chunk is not marked done

    before = list(seen)
    monkeypatch.setattr(gs, "ollama_chat",
                        lambda model, prompt, url, timeout=180: json.dumps([]))
    monkeypatch.setattr(sys, "argv", ["gen_synthetic.py", "--out-dir", str(tmp_path), "--target", "99"])
    gs.main()
    # the two completed chunks were not re-sent
    assert seen == before
