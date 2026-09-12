"""Merge adapter -> fp16 HF -> GGUF (Q4_K_M, Q8_0) -> Ollama Modelfile.
Run on the desktop. Requires llama.cpp cloned at --llama-cpp."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import torch

from scripts import common

# Ollama Go template that reproduces the Qwen3 chat format used at training time
# (see scripts/common.py:to_prompt_completion). ASSISTANT_PREFIX is a literal
# placeholder substituted with the exact generation-prompt suffix the training
# tokenizer emits (e.g. "<|im_start|>assistant\n<think>\n\n</think>\n\n" for
# Qwen3 non-thinking), so Ollama's inference-time prompt matches training exactly.
_MODELFILE_TEMPLATE = '''TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}
{{- end }}ASSISTANT_PREFIX"""
'''


def render_modelfile(gguf_name: str, system_prompt: str, assistant_prefix: str) -> str:
    template = _MODELFILE_TEMPLATE.replace("ASSISTANT_PREFIX", assistant_prefix)
    return (
        f"FROM ./{gguf_name}\n"
        f'SYSTEM """{system_prompt}"""\n'
        f"{template}"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER temperature 0.6\n"
        "PARAMETER stop \"<|im_end|>\"\n"
    )


def preflight(llama_cpp: str) -> tuple[str, str]:
    """Verify llama.cpp is cloned and built before spending time on the merge.

    Returns (convert_script, quant_bin). Raises SystemExit with a clear
    message if either path is missing.
    """
    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    quant_bin = os.path.join(llama_cpp, "build", "bin", "llama-quantize")
    missing = [p for p in (convert_script, quant_bin) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "llama.cpp not ready at "
            f"{llama_cpp!r}, missing: {', '.join(missing)}. "
            "Clone and build llama.cpp first (see README)."
        )
    return convert_script, quant_bin


def _merge(model_name, adapter, out_dir):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--llama-cpp", default="../llama.cpp")
    ap.add_argument("--name", default="decria-sec")
    args = ap.parse_args()
    convert_script, quant_bin = preflight(args.llama_cpp)

    cfg = common.load_config(args.config)
    run = cfg["output_dir"]
    adapter = os.path.join(run, "adapter")
    merged = os.path.join(run, "merged")
    gguf_dir = os.path.join(run, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    _merge(cfg["model_name"], adapter, merged)

    f16 = os.path.join(gguf_dir, f"{args.name}-f16.gguf")
    subprocess.run([sys.executable, convert_script,
                    merged, "--outfile", f16, "--outtype", "f16"], check=True)
    outputs = {}
    for qt in ["Q4_K_M", "Q8_0"]:
        dst = os.path.join(gguf_dir, f"{args.name}-{qt.lower()}.gguf")
        subprocess.run([quant_bin, f16, dst, qt], check=True)
        outputs[qt] = dst

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(adapter)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    assistant_prefix = rendered.split("<|im_end|>\n")[-1]
    assert assistant_prefix.startswith("<|im_start|>assistant"), \
        f"unexpected assistant prefix from tokenizer: {assistant_prefix!r}"

    mf = render_modelfile(os.path.basename(outputs["Q4_K_M"]), common.SYSTEM_PROMPT, assistant_prefix)
    with open(os.path.join(gguf_dir, "Modelfile"), "w", encoding="utf-8") as fh:
        fh.write(mf)
    with open("Modelfile", "w", encoding="utf-8") as fh:
        fh.write(mf)
    print("GGUF written:", outputs)
    print(f"Next: cd {gguf_dir} && ollama create {args.name} -f Modelfile")


if __name__ == "__main__":
    main()
