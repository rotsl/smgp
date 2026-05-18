"""Tests for the SpectralAttention module."""
import numpy as np
import pytest

from smgp.attention.spectral_attn import SpectralAttention
from smgp.core.graph import SpectralMemoryGraph


@pytest.fixture
def attention():
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    return SpectralAttention(g, hidden_dim=64, num_heads=4, num_scales=2)


class TestSpectralAttention:
    def test_build_graph_from_tokens(self, attention):
        tokens = np.random.randn(10, 32) * 0.1
        graph = attention.build_graph_from_tokens(tokens)
        assert graph.num_nodes == 10

    def test_forward(self, attention):
        tokens = np.random.randn(10, 64) * 0.1
        attention.build_graph_from_tokens(tokens)
        output = attention.forward(tokens)
        assert output.shape == (10, 64)
        assert not np.all(output == 0)

    def test_forward_empty(self, attention):
        empty = np.zeros((0, 64))
        output = attention.forward(empty)
        assert output.shape == (0, 64)

    def test_forward_single_token(self, attention):
        tokens = np.random.randn(1, 64) * 0.1
        attention.build_graph_from_tokens(tokens)
        output = attention.forward(tokens)
        assert output.shape == (1, 64)

    def test_hierarchical_coarsening(self, attention):
        tokens = np.random.randn(16, 64) * 0.1
        attention.build_graph_from_tokens(tokens)
        levels = attention.hierarchical_coarsening()
        assert len(levels) >= 2  # At least original + 1 coarsened
        assert levels[0].num_nodes == 16
        # Coarsened level should have fewer nodes
        if len(levels) > 1:
            assert levels[1].num_nodes <= levels[0].num_nodes

    def test_forward_deterministic(self, attention):
        tokens = np.random.randn(8, 64) * 0.1
        attention.build_graph_from_tokens(tokens)
        output1 = attention.forward(tokens)
        output2 = attention.forward(tokens)
        np.testing.assert_array_equal(output1, output2)
