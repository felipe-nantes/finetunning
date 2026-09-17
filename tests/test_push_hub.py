import json

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
    manifest = {"sources": [{"repo": "Trendyol/x", "source": "trendyol", "license": "apache-2.0",
                             "kept": 2500}],
                "synthetic": {"by_source": {"synthetic-attack": 400}}, "train": 3600, "val": 190}
    counts = {"trendyol": 2500, "synthetic-attack": 400, "synthetic-oos": 98}
    card = ph.render_dataset_card(manifest, counts)
    assert "cc-by-sa-4.0" in card
    assert "Trendyol/x" in card
    assert "2500 rows" in card
    assert "synthetic-attack: 400" in card
    assert "synthetic-oos" in card and "98" in card
    assert "evaluated and excluded" not in card   # nothing excluded -> no section


def test_count_sources_reads_the_published_files(tmp_path):
    rows = [{"source": "trendyol"}, {"source": "trendyol"}, {"source": "synthetic-wstg"}]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "val.jsonl").write_text(json.dumps({"source": "trendyol"}) + "\n\n")
    counts = ph.count_sources([str(tmp_path / "train.jsonl"), str(tmp_path / "val.jsonl")])
    assert counts == {"trendyol": 3, "synthetic-wstg": 1}


def _overlapped_manifest():
    # Regression: a source fully overlapped by another one has a large pre-dedup
    # `kept` in the manifest but contributes zero rows to train/val. The cards
    # must not present it as a contributor.
    return {"sources": [{"repo": "Trendyol/x", "source": "trendyol", "license": "apache-2.0",
                         "kept": 52382},
                        {"repo": "Other/dupe", "source": "dupe", "license": "apache-2.0",
                         "kept": 2390}],
            "excluded_sources": [{"repo": "Old/dropped", "reason": "exact duplicate of Trendyol/x",
                                  "kept_pre_dedup": 2390, "kept_post_dedup": 0}],
            "train": 2850, "val": 150}


def test_dataset_card_ignores_manifest_kept_for_zero_row_sources():
    counts = {"trendyol": 3000}
    card = ph.render_dataset_card(_overlapped_manifest(), counts)
    assert "Other/dupe" not in card
    assert "apache-2.0 — 2390 rows" not in card       # the pre-dedup `kept` never becomes a contribution line
    assert "Trendyol/x` — apache-2.0 — 3000 rows" in card
    assert "## Sources evaluated and excluded" in card
    assert "Old/dropped" in card and "exact duplicate of Trendyol/x" in card
    assert "kept 2390 rows before dedup, 0 after" in card


def test_model_card_ignores_manifest_kept_for_zero_row_sources():
    meta = {"model_name": "Qwen/Qwen3-0.6B", "base_license": "apache-2.0",
            "hours": 3.0, "gpu": "GTX 1060 3GB",
            "eval": {"base": {"mcq_acc": 0.5}, "adapter": {"mcq_acc": 0.5}},
            "manifest": _overlapped_manifest(), "source_counts": {"trendyol": 3000},
            "lora_r": 16, "effective_batch": 16, "epochs": 2}
    card = ph.render_model_card(meta)
    assert "Other/dupe" not in card
    assert "Trendyol/x` — apache-2.0 — 3000 rows (public)" in card
    assert "Old/dropped" in card and "0 after" in card


def test_render_model_card_full_spec_sections():
    meta = {"model_name": "Qwen/Qwen3-0.6B", "base_license": "apache-2.0",
            "hours": 3.4, "gpu": "GTX 1060 3GB",
            "eval": {"base": {"mcq_acc": 0.61, "ci95": [0.57, 0.65], "val_loss": 1.42},
                     "adapter": {"mcq_acc": 0.64, "ci95": [0.60, 0.68], "val_loss": 1.20}},
            "manifest": {"sources": [{"repo": "Trendyol/x", "source": "trendyol",
                                      "license": "apache-2.0", "kept": 2500}],
                         "synthetic": {"by_source": {"synthetic-attack": 400}},
                         "oos_refusals": {"added": 98}},
            "source_counts": {"trendyol": 2500, "synthetic-attack": 400, "synthetic-oos": 98},
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
