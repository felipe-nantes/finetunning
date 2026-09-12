from scripts import push_hub as ph


def test_render_model_card_includes_metrics_and_license():
    meta = {"model_name": "Qwen/Qwen3-1.7B", "base_license": "apache-2.0",
            "hours": 9.5, "gpu": "GTX 1060 6GB",
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
