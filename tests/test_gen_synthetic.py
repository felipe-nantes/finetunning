import json

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
    en = gs.build_gen_prompt("chunk text", lang="en")
    assert "json" in en.lower()


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


def test_update_manifest_accumulates(tmp_path):
    path = str(tmp_path / "manifest.json")
    gs._update_manifest(
        [{"source": "synthetic-attack"}, {"source": "synthetic-attack"}], "model-a", path=path
    )
    gs._update_manifest([{"source": "synthetic-cwe"}], "model-b", path=path)

    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    assert man["synthetic"]["added"] == 3
    assert man["synthetic"]["by_source"] == {"synthetic-attack": 2, "synthetic-cwe": 1}
