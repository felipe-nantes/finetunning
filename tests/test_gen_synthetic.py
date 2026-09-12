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
