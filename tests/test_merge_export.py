import pytest

from scripts import merge_export as me


def test_render_modelfile_has_template_system_and_params():
    prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    mf = me.render_modelfile("m.gguf", "SYS", prefix)
    assert 'TEMPLATE """' in mf
    assert "{{ .System }}" in mf
    assert prefix in mf
    assert "FROM ./m.gguf" in mf
    assert "num_ctx 4096" in mf
    assert "temperature 0.6" in mf
    assert 'stop "<|im_end|>"' in mf


def test_preflight_raises_when_llama_cpp_missing(tmp_path):
    with pytest.raises(SystemExit):
        me.preflight(str(tmp_path / "no-llama-cpp"))


def test_preflight_returns_paths_when_present(tmp_path):
    convert = tmp_path / "convert_hf_to_gguf.py"
    convert.write_text("# stub")
    quant_dir = tmp_path / "build" / "bin"
    quant_dir.mkdir(parents=True)
    quant_bin = quant_dir / "llama-quantize"
    quant_bin.write_text("stub")

    got_convert, got_quant = me.preflight(str(tmp_path))
    assert got_convert == str(convert)
    assert got_quant == str(quant_bin)
