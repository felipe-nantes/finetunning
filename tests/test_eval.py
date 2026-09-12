from scripts import eval_model as ev


def test_parse_cybermetric():
    obj = {"questions": [
        {"question": "Q1", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"}, "solution": "B"},
    ]}
    items = ev.parse_cybermetric(obj)
    assert items[0]["answer"] == "B"
    assert items[0]["choices"]["C"] == "c"


def test_score_choice_argmax():
    assert ev.score_choice({"A": -3.1, "B": -0.2, "C": -5.0, "D": -1.0}) == "B"


def test_bootstrap_ci_bounds():
    flags = [1] * 80 + [0] * 20
    acc, lo, hi = ev.bootstrap_ci(flags, seed=0)
    assert abs(acc - 0.8) < 1e-9
    assert 0.6 < lo < 0.8 < hi < 0.95


import json


def test_frozen_prompt_files_shape():
    q = [json.loads(l) for l in open("eval/prompts.jsonl", encoding="utf-8") if l.strip()]
    r = [json.loads(l) for l in open("eval/regression_prompts.jsonl", encoding="utf-8") if l.strip()]
    assert len(q) == 12 and len(r) == 5
    assert all({"lang", "theme", "prompt"} <= set(p) for p in q)
    assert all({"lang", "prompt"} <= set(p) for p in r)
    assert sum(p["lang"] == "pt" for p in q) == 4
    assert any(p["theme"] == "outofscope" for p in q)
