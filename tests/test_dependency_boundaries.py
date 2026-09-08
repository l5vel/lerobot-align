"""Positive controls and real model-loading coverage for advisory boundaries."""

import pytest

from tests.dependency_boundaries import rejected_advisory_operations, watched_operations


def test_every_tripwire_rejects_calls_and_restores_the_original():
    operations = watched_operations()
    originals = [getattr(owner, attribute) for owner, attribute, _ in operations]
    with rejected_advisory_operations() as attempts:
        for owner, attribute, name in operations:
            with pytest.raises(AssertionError, match="Dependency advisory operation reached"):
                getattr(owner, attribute)(None)
            assert attempts[-1] == name
    assert len(attempts) == 14
    for (owner, attribute, _), original in zip(operations, originals, strict=True):
        assert getattr(owner, attribute) is original


@pytest.fixture
def tiny_local_bert(tmp_path, monkeypatch):
    """Original random weights and vocabulary; no download or third-party data."""
    from transformers import BertConfig, BertModel, BertTokenizer

    root = tmp_path / "tiny-bert"
    root.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "pick", "up", "cup", "place", "bottle"]
    (root / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
    tokenizer = BertTokenizer(vocab=str(root / "vocab.txt"))
    tokenizer.save_pretrained(root)
    BertModel(BertConfig(vocab_size=len(vocabulary), hidden_size=8, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=16, max_position_embeddings=64)).save_pretrained(root, max_shard_size=200)
    assert (root / "model.safetensors.index.json").exists()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    return root


def test_embedding_evaluation_loads_and_encodes_without_affected_operations(tiny_local_bert):
    from evaluation.metrics.semantic import EmbeddingMatcher

    with rejected_advisory_operations() as attempts:
        matcher = EmbeddingMatcher(model_name=str(tiny_local_bert))
        matcher.warm(["pick up cup", "place bottle"])
        assert matcher.similarity("pick up cup", "pick up cup") == pytest.approx(1.0, abs=1e-5)
        assert len(matcher._cache) == 2
        assert not attempts


def test_transformers_server_loads_processor_and_model_without_affected_operations(tiny_local_bert):
    from transformers.cli.serving.model_manager import ModelManager

    with rejected_advisory_operations() as attempts:
        manager = ModelManager(device="cpu", dtype="float32")
        assert manager.trust_remote_code is False
        processor = manager._load_processor(f"{tiny_local_bert}@local")
        model = manager._load_model(f"{tiny_local_bert}@local")
        inputs = processor("pick up cup", return_tensors="pt")
        assert model(**inputs).last_hidden_state.shape[-1] == 8
        assert not attempts
