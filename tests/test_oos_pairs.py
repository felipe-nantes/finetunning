from scripts import oos_pairs
from scripts import build_dataset as bd
from scripts import common


def test_build_oos_rows_shape_and_balance():
    rows = oos_pairs.build_oos_rows()
    assert len(rows) >= 90                       # target ~100-150 pairs
    assert sum(r["lang"] == "en" for r in rows) == sum(r["lang"] == "pt" for r in rows)
    users = [r["messages"][1]["content"] for r in rows]
    assert len(set(users)) == len(users)         # no duplicate prompts
    for r in rows:
        assert r["source"] == "synthetic-oos"
        assert r["messages"][0]["content"] == common.SYSTEM_PROMPT
        assert r["messages"][-1]["role"] == "assistant"


def test_oos_answers_are_scoped_redirects_not_ai_disclaimers():
    for r in oos_pairs.build_oos_rows():
        answer = r["messages"][-1]["content"].lower()
        # a trained refusal that redirects to the authorized angle, never the
        # base model's "as an AI language model" disclaimer
        assert "as an ai language model" not in answer
        assert "authorized" in answer or "autorizado" in answer
        assert "scope" in answer or "escopo" in answer


class _FakeTok:
    eos_token = "<eos>"

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False):
        return " ".join(m["content"] for m in msgs) + " ASSISTANT:"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def test_oos_rows_survive_the_token_budget_filter():
    kept, stats = bd.filter_by_token_budget(oos_pairs.build_oos_rows(), _FakeTok(),
                                            max_length=768, min_completion_tokens=64)
    assert len(kept) == stats["seen"]            # none dropped: not too long, not too short
    assert stats["over_max_length"] == 0
