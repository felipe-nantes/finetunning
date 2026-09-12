from scripts import build_dataset as bd  # module aliased; see note in Step 3


def test_normalize_row_maps_schema():
    raw = {"system": "S", "user": "How to enumerate ports with nmap for recon?",
           "assistant": "word " * 120}
    row = bd.normalize_row(raw, "trendyol")
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["content"].startswith("How to enumerate")
    assert row["messages"][2]["role"] == "assistant"
    assert row["source"] == "trendyol"
    assert row["lang"] == "en"
    assert row["theme"] == "recon"


def test_normalize_row_rejects_short_and_refusal():
    assert bd.normalize_row({"system": "S", "user": "u", "assistant": "no"}, "x") is None
    assert bd.normalize_row(
        {"system": "S", "user": "u " * 50, "assistant": "I cannot help with that. " * 20}, "x"
    ) is None


def test_stratified_cap_balances_themes():
    rows = ([{"theme": "web"}] * 100) + ([{"theme": "recon"}] * 10)
    out = bd.stratified_cap(rows, cap=40, key="theme", seed=1)
    n_web = sum(1 for r in out if r["theme"] == "web")
    n_recon = sum(1 for r in out if r["theme"] == "recon")
    assert len(out) == 40
    assert n_recon >= 10  # small class fully kept
    assert n_web <= 30
