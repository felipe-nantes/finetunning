import torch

from scripts import train


def test_build_lora_config_targets():
    cfg = {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
    lc = train.build_lora_config(cfg)
    assert lc.r == 16 and lc.lora_alpha == 32
    for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        assert m in lc.target_modules
    assert lc.bias == "none"
    assert lc.task_type == "CAUSAL_LM"


def test_render_dataset_produces_prompt_completion():
    from datasets import Dataset

    class FakeTok:
        eos_token = "<|end|>"
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
            return "P:" + msgs[-1]["content"]
    ds = Dataset.from_list([{"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ], "source": "x", "lang": "en", "theme": "web"}])
    out = train.render_dataset(ds, FakeTok())
    assert set(out.column_names) == {"prompt", "completion"}
    assert out[0]["completion"] == "a<|end|>"


def test_build_sft_config_pascal_safe(tmp_path):
    cfg = {"output_dir": str(tmp_path / "out"), "max_length": 768,
           "per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
           "learning_rate": 2e-4, "num_train_epochs": 2, "max_steps": -1,
           "logging_steps": 10, "save_steps": 200, "eval_steps": 200, "seed": 42}
    sft = train.build_sft_config(cfg)
    assert sft.fp16 is False and sft.bf16 is False
    assert sft.completion_only_loss is True
    assert sft.max_length == 768
    assert sft.optim == "adamw_torch"
    assert sft.lr_scheduler_type == "cosine"
    assert sft.gradient_checkpointing is True
    assert sft.per_device_eval_batch_size == 1


def test_last_checkpoint_ignores_non_numeric_entries(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-10" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-200").mkdir()  # partial save: no trainer_state.json
    (tmp_path / "checkpoint-notes.txt").write_text("not a checkpoint dir")
    (tmp_path / "checkpoint-final").mkdir()
    assert train._last_checkpoint(str(tmp_path)).endswith("checkpoint-10")
    assert train._last_checkpoint(str(tmp_path / "missing")) is None


def test_force_fp32_trainable_casts_only_trainable_bf16_params():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    model = model.to(torch.bfloat16)
    model[1].requires_grad_(False)

    n_cast = train.force_fp32_trainable(model)

    assert n_cast == 2  # layer 0 weight + bias
    assert model[0].weight.dtype == torch.float32
    assert model[0].bias.dtype == torch.float32
    assert model[1].weight.dtype == torch.bfloat16
    assert model[1].bias.dtype == torch.bfloat16
