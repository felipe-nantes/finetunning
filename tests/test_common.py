import pytest
from scripts import common


def test_system_prompt_scope():
    p = common.SYSTEM_PROMPT.lower()
    assert "authorized" in p or "autoriz" in p


def test_detect_lang_en_pt():
    assert common.detect_lang("How do I enumerate open ports with nmap?") == "en"
    assert common.detect_lang("Como faço um reconhecimento passivo do alvo?") == "pt"


def test_response_len_ok_bounds():
    assert not common.response_len_ok("too short")
    assert common.response_len_ok("word " * 100)
    assert not common.response_len_ok("word " * 5000)


def test_is_refusal():
    assert common.is_refusal("I cannot help with that request.")
    assert common.is_refusal("As an AI language model, I'm unable to assist.")
    assert not common.is_refusal("Use nmap -sV to fingerprint services.")


def test_tag_theme():
    assert common.tag_theme("Run nmap and enumerate subdomains for recon") == "recon"
    assert common.tag_theme("Reflected XSS in the search parameter") == "web"
    assert common.tag_theme("linux privilege escalation via SUID binary") == "privesc"
    assert common.tag_theme("detect lateral movement, blue team monitoring") == "defense"


def test_exact_dedup():
    rows = [{"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]}]
    assert len(common.exact_dedup(rows)) == 2


def test_minhash_dedup_removes_near_dupes():
    base = "how to test a login form for sql injection with sqlmap step by step"
    rows = [
        {"messages": [{"role": "user", "content": base}]},
        {"messages": [{"role": "user", "content": base + " please"}]},
        {"messages": [{"role": "user", "content": "explain the tcp three way handshake"}]},
    ]
    out = common.minhash_dedup(rows, threshold=0.7)
    assert len(out) == 2


def test_minhash_dedup_ignores_shared_system_prompt():
    # Every row in the pipeline carries the identical SYSTEM_PROMPT verbatim.
    # If _row_text included it, two rows about genuinely different topics but
    # written in similar templated phrasing (common across the public
    # cybersecurity datasets) can collide: 0.77 jaccard without the system
    # text, 0.87 with it -- crossing the 0.8 threshold only because of the
    # shared boilerplate, not because the content is a near-duplicate.
    row_a = {"messages": [
        {"role": "system", "content": common.SYSTEM_PROMPT},
        {"role": "user", "content": "What do I check during an authorized pentest for injection issues in the login form?"},
        {"role": "assistant", "content": (
            "During an authorized penetration testing engagement, check the login "
            "form for injection issues, review the relevant configuration, confirm "
            "the finding with a safe proof of concept, and document the result for "
            "the client report before moving to the next target."
        )},
    ]}
    row_b = {"messages": [
        {"role": "system", "content": common.SYSTEM_PROMPT},
        {"role": "user", "content": "What do I check during an authorized pentest for privilege escalation issues in the cron jobs?"},
        {"role": "assistant", "content": (
            "During an authorized penetration testing engagement, check the cron "
            "jobs for privilege escalation issues, review the relevant permissions, "
            "confirm the finding with a safe proof of concept, and document the "
            "result for the client report before moving to the next target."
        )},
    ]}
    out = common.minhash_dedup([row_a, row_b], threshold=0.8)
    assert len(out) == 2


def test_to_prompt_completion_shapes():
    class FakeTok:
        eos_token = "<|im_end|>"
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
            assert tokenize is False
            assert add_generation_prompt is True
            return "".join(m["content"] for m in msgs) + "|GEN|"
    ex = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]}
    out = common.to_prompt_completion(ex, FakeTok())
    assert out["prompt"] == "sysu|GEN|"
    assert out["completion"] == "a<|im_end|>"
