"""Gate: verify the desktop GPU can actually run QLoRA before any long run.
Run on the desktop inside WSL2:  uv run python scripts/00_check_env.py
"""
import sys
import time
import torch


def main() -> int:
    ok = True

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available"); return 1
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    print(f"GPU: {name}  capability sm_{cap[0]}{cap[1]}")
    print(f"torch {torch.__version__}  arch_list={arch_list}")

    has_pascal = any(a in arch_list for a in ("sm_60", "sm_61"))
    if not has_pascal:
        print("FAIL: wheel has no sm_60/sm_61; wrong CUDA build (need cu126, not cu130+)")
        ok = False

    # NF4 round trip on a real small model + forward/backward
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_id = "Qwen/Qwen3-0.6B"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, dtype=torch.float32, device_map={"": 0},
        )
    except Exception as e:
        print(f"FAIL: 4-bit load raised: {e}"); return 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    model.gradient_checkpointing_enable()
    model.train()

    torch.cuda.reset_peak_memory_stats()
    seq = 768
    ids = torch.randint(0, tok.vocab_size, (1, seq), device="cuda")
    t0 = time.time()
    steps = 20
    for _ in range(steps):
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    toks_per_s = steps * seq / dt
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"throughput: {toks_per_s:.0f} tok/s   peak VRAM: {peak_gb:.2f} GB   loss={out.loss.item():.3f}")

    if not torch.isfinite(out.loss):
        print("FAIL: loss is not finite"); ok = False
    if toks_per_s < 40:
        print("WARN: < 40 tok/s. Training will be very slow; consider seq 512 / 1 epoch / plan B.")
    if peak_gb > 5.5:
        print("WARN: peak VRAM > 5.5 GB on 0.6B; 1.7B may OOM. Reduce seq.")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
