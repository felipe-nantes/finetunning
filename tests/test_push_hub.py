from scripts import push_hub as ph


def test_render_model_card_includes_metrics_and_license():
    meta = {"model_name": "Qwen/Qwen3-1.7B", "base_license": "apache-2.0",
            "hours": 9.5, "gpu": "GTX 1060 6GB",
            "eval": {"base": {"mcq_acc": 0.61}, "adapter": {"mcq_acc": 0.64}}}
    card = ph.render_model_card(meta)
    assert "apache-2.0" in card
    assert "0.64" in card
    assert "GTX 1060" in card
    assert "authorized" in card.lower()


def test_render_dataset_card_lists_sources():
    manifest = {"sources": [{"repo": "Trendyol/x", "license": "apache-2.0", "kept": 2500}],
                "synthetic": {"by_source": {"synthetic-attack": 400}}, "train": 3600, "val": 190}
    card = ph.render_dataset_card(manifest)
    assert "cc-by-sa-4.0" in card
    assert "Trendyol/x" in card
