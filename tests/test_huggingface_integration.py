"""Tests for the HuggingFace integration module."""
import numpy as np
import pytest

from smgp.integration.huggingface import SMGPConfig, SMGPForCausalLM


@pytest.fixture
def model():
    config = SMGPConfig(
        hd_dim=100,
        hidden_dim=64,
        num_heads=4,
        vocab_size=100,
        max_position_embeddings=32,
    )
    return SMGPForCausalLM(config)


class TestSMGPConfig:
    def test_defaults(self):
        config = SMGPConfig()
        assert config.hd_dim == 10000
        assert config.hidden_dim == 256

    def test_custom(self):
        config = SMGPConfig(hd_dim=100, hidden_dim=64)
        assert config.hd_dim == 100
        assert config.hidden_dim == 64

    def test_to_dict(self):
        config = SMGPConfig(hd_dim=100)
        d = config.to_dict()
        assert d["hd_dim"] == 100


class TestSMGPForCausalLM:
    def test_init(self, model):
        assert model.config.hd_dim == 100
        assert model.config.vocab_size == 100

    def test_forward(self, model):
        input_ids = np.random.randint(0, 100, size=(1, 8))
        output = model.forward(input_ids)
        assert "logits" in output
        assert output["logits"].shape[0] == 1
        assert output["logits"].shape[1] == 8
        assert output["logits"].shape[2] == 100

    def test_forward_1d(self, model):
        input_ids = np.random.randint(0, 100, size=8)
        output = model.forward(input_ids)
        assert output["logits"].shape == (1, 8, 100)

    def test_generate(self, model):
        input_ids = np.random.randint(0, 100, size=5)
        generated = model.generate(input_ids, max_length=10, temperature=0.5)
        assert generated.shape[0] == 15  # 5 input + 10 generated

    def test_save_load(self, model, tmp_path):
        save_dir = str(tmp_path / "test_model")
        model.save_pretrained(save_dir)
        loaded = SMGPForCausalLM.from_pretrained(save_dir)
        assert loaded.config.hd_dim == model.config.hd_dim
        assert loaded.config.vocab_size == model.config.vocab_size
        np.testing.assert_array_equal(loaded.embeddings, model.embeddings)
