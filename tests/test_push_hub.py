from scripts import push_hub as ph


def test_render_model_card_includes_metrics_and_license():
    meta = {"model_name": "Qwen/Qwen3-1.7B", "base_license": "apache-2.0",
            "hours": 9.5, "gpu": "GTX 1060 3GB",
            "eval": {"base": {"mcq_acc": 0.61}, "adapter": {"mcq_acc": 0.64}},
            "lora_r": 8, "effective_batch": 4, "epochs": 1}
    card = ph.render_model_card(meta)
    assert "apache-2.0" in card
    assert "| base | 0.610 |" in card
    assert "| adapter | 0.640 |" in card
    assert "GTX 1060" in card
    assert "authorized" in card.lower()
    assert "r=8" in card
    assert "effective batch 4" in card
    assert "1 epochs" in card


def test_render_dataset_card_lists_sources():
    manifest = {"sources": [{"repo": "Trendyol/x", "license": "apache-2.0", "kept": 2500}],
                "synthetic": {"by_source": {"synthetic-attack": 400}}, "train": 3600, "val": 190}
    card = ph.render_dataset_card(manifest)
    assert "cc-by-sa-4.0" in card
    assert "Trendyol/x" in card
    assert "2500 rows" in card
    assert "synthetic-attack: 400" in card


def test_render_model_card_full_spec_sections():
    meta = {"model_name": "Qwen/Qwen3-0.6B", "base_license": "apache-2.0",
            "hours": 3.4, "gpu": "GTX 1060 3GB",
            "eval": {"base": {"mcq_acc": 0.61, "ci95": [0.57, 0.65], "val_loss": 1.42},
                     "adapter": {"mcq_acc": 0.64, "ci95": [0.60, 0.68], "val_loss": 1.20}},
            "manifest": {"sources": [{"repo": "Trendyol/x", "license": "apache-2.0", "kept": 2500}],
                         "synthetic": {"by_source": {"synthetic-attack": 400}},
                         "oos_refusals": {"added": 98}},
            "samples_md": "## Explain IDOR\n\n**base:** ...\n\n**adapter:** ...",
            "lora_r": 16, "lora_alpha": 32, "max_length": 768,
            "effective_batch": 16, "epochs": 2}
    card = ph.render_model_card(meta)
    assert "library_name: peft" in card              # spec (d)
    assert "GTX 1060 3GB" in card                     # hardware correction
    assert "[0.570, 0.650]" in card and "[0.600, 0.680]" in card  # 95% CI
    assert "1.420" in card and "1.200" in card        # val loss
    assert "Trendyol/x" in card                        # sources + licenses
    assert "synthetic-oos" in card                     # refusal pairs credited
    assert "Limitations" in card
    assert "unintended use" in card.lower()
    assert "Explain IDOR" in card                      # qualitative samples embedded


def test_render_model_card_tolerates_mcq_only_eval():
    meta = {"model_name": "Qwen/Qwen3-0.6B", "base_license": "apache-2.0",
            "hours": 3.0, "gpu": "GTX 1060 3GB",
            "eval": {"base": {"mcq_acc": 0.5}, "adapter": {"mcq_acc": 0.5}},
            "lora_r": 16, "effective_batch": 16, "epochs": 2}
    card = ph.render_model_card(meta)  # no manifest/samples/ci/val_loss
    assert "| base | 0.500 | — | — |" in card


def test_samples_excerpt_missing_file_is_empty(tmp_path):
    assert ph._samples_excerpt(str(tmp_path / "nope.md")) == ""


def test_samples_excerpt_takes_first_blocks(tmp_path):
    p = tmp_path / "samples.md"
    p.write_text("# Sample outputs\n\n## Q1\n\nbody1\n\n## Q2\n\nbody2\n\n## Q3\n\nbody3\n\n## Q4\n\nbody4\n",
                 encoding="utf-8")
    out = ph._samples_excerpt(str(p), max_prompts=2)
    assert "## Q1" in out and "## Q2" in out
    assert "Q3" not in out
