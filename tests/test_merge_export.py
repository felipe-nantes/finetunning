from scripts import merge_export as me


def test_render_modelfile_has_from_and_system():
    mf = me.render_modelfile("decria-sec-q4_k_m.gguf", "SYS PROMPT")
    assert "FROM ./decria-sec-q4_k_m.gguf" in mf
    assert "SYS PROMPT" in mf
    assert "num_ctx 4096" in mf
