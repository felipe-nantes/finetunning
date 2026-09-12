"""Merge adapter -> fp16 HF -> GGUF (Q4_K_M, Q8_0) -> Ollama Modelfile.
Run on the desktop. Requires llama.cpp cloned at --llama-cpp."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import torch

from scripts import common


def render_modelfile(gguf_name: str, system_prompt: str) -> str:
    return (
        f"FROM ./{gguf_name}\n"
        f'SYSTEM """{system_prompt}"""\n'
        "PARAMETER num_ctx 4096\n"
        "PARAMETER temperature 0.6\n"
        "PARAMETER stop \"<|im_end|>\"\n"
    )


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
    cfg = common.load_config(args.config)
    run = cfg["output_dir"]
    adapter = os.path.join(run, "adapter")
    merged = os.path.join(run, "merged")
    gguf_dir = os.path.join(run, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    _merge(cfg["model_name"], adapter, merged)

    f16 = os.path.join(gguf_dir, f"{args.name}-f16.gguf")
    subprocess.run([sys.executable, os.path.join(args.llama_cpp, "convert_hf_to_gguf.py"),
                    merged, "--outfile", f16, "--outtype", "f16"], check=True)
    quant = os.path.join(args.llama_cpp, "build", "bin", "llama-quantize")
    outputs = {}
    for qt in ["Q4_K_M", "Q8_0"]:
        dst = os.path.join(gguf_dir, f"{args.name}-{qt.lower()}.gguf")
        subprocess.run([quant, f16, dst, qt], check=True)
        outputs[qt] = dst

    mf = render_modelfile(os.path.basename(outputs["Q4_K_M"]), common.SYSTEM_PROMPT)
    open(os.path.join(gguf_dir, "Modelfile"), "w", encoding="utf-8").write(mf)
    open("Modelfile", "w", encoding="utf-8").write(mf)
    print("GGUF written:", outputs)
    print(f"Next: cd {gguf_dir} && ollama create {args.name} -f Modelfile")


if __name__ == "__main__":
    main()
