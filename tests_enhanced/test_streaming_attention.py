"""Tests for :class:`StreamingSpectralAttention`.

Verifies that a 64-token sequence processed with ``chunk_size=32`` produces
output of the correct shape and contains finite values.
"""
from __future__ import annotations

import numpy as np
import pytest

from smgp.attention.streaming_attention import StreamingSpectralAttention
from smgp.core.graph import SpectralMemoryGraph


class TestStreamingSpectralAttention:
    """Tests for the streaming spectral attention mechanism."""

    @pytest.fixture()
    def graph(self) -> SpectralMemoryGraph:
        return SpectralMemoryGraph(hd_dim=100, seed=42)

    @pytest.fixture()
    def attn(self, graph: SpectralMemoryGraph) -> StreamingSpectralAttention:
        return StreamingSpectralAttention(
            graph=graph,
            hidden_dim=64,
            num_heads=4,
            num_scales=2,
            chunk_size=32,
            overlap=8,
        )

    def test_output_shape(self, attn: StreamingSpectralAttention) -> None:
        """Output shape must be (N, hidden_dim)."""
        n_tokens = 64
        dim = 64
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((n_tokens, dim))

        output = attn.forward(embeddings)

        assert output.shape == (n_tokens, dim), (
            f"Expected shape ({n_tokens}, {dim}), got {output.shape}"
        )

    def test_output_finite(self, attn: StreamingSpectralAttention) -> None:
        """All output values must be finite (no NaN / Inf)."""
        rng = np.random.default_rng(1)
        embeddings = rng.standard_normal((64, 64))

        output = attn.forward(embeddings)

        assert np.all(np.isfinite(output)), "Output contains NaN or Inf values"

    def test_reset_clears_state(self, attn: StreamingSpectralAttention) -> None:
        """After reset the token_offset should be zero."""
        rng = np.random.default_rng(2)
        embeddings = rng.standard_normal((32, 64))
        attn.forward(embeddings)

        attn.reset()
        state = attn.get_state()
        assert state["token_offset"] == 0

    def test_chunk_override(self, attn: StreamingSpectralAttention) -> None:
        """Passing a custom chunk_size should still produce correct shape."""
        rng = np.random.default_rng(3)
        embeddings = rng.standard_normal((64, 64))

        # Use a different chunk size at call time
        output = attn.forward(embeddings, chunk_size=16)
        assert output.shape == (64, 64)

    def test_stream_state_restore(self, attn: StreamingSpectralAttention) -> None:
        """Providing a stream_state dict should restore offset."""
        rng = np.random.default_rng(4)
        embeddings = rng.standard_normal((32, 64))

        state = {"token_offset": 100}
        output = attn.forward(embeddings, stream_state=state)
        assert output.shape == (32, 64)
        assert attn.get_state()["token_offset"] == 100

    def test_single_chunk(self, graph: SpectralMemoryGraph) -> None:
        """When the sequence fits in a single chunk output is straightforward."""
        attn = StreamingSpectralAttention(
            graph=graph,
            hidden_dim=32,
            num_heads=2,
            num_scales=1,
            chunk_size=128,
            overlap=0,
        )
        rng = np.random.default_rng(5)
        embeddings = rng.standard_normal((16, 32))

        output = attn.forward(embeddings)
        assert output.shape == (16, 32)
        assert np.all(np.isfinite(output))
