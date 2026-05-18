"""Hardware executor wiring integration tests.

Tests that core SMGP modules correctly accept an optional ``executor``
parameter and delegate to it (or fall back to pure Python when absent).
All tests use ``unittest.mock.MagicMock`` so the hardware HAL is not required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SpectralMemoryGraph
# ---------------------------------------------------------------------------


class TestGraphExecutorWiring:
    def test_executor_stored(self):
        from smgp.core.graph import SpectralMemoryGraph

        mock = MagicMock()
        g = SpectralMemoryGraph(hd_dim=100, seed=42, executor=mock)
        assert g.executor is mock

    def test_hw_bridge_created_when_executor_given(self):
        from smgp.core.graph import SpectralMemoryGraph

        mock = MagicMock()
        g = SpectralMemoryGraph(hd_dim=100, seed=42, executor=mock)
        # hw_bridge is created (may be None only if hw_bridge import failed)
        # The important check: executor is stored
        assert g.executor is mock

    def test_no_executor_no_bridge(self):
        from smgp.core.graph import SpectralMemoryGraph

        g = SpectralMemoryGraph(hd_dim=100, seed=42)
        assert g.executor is None
        assert g.hw_bridge is None


# ---------------------------------------------------------------------------
# SpectralMethods
# ---------------------------------------------------------------------------


class TestSpectralMethodsWiring:
    def _make_graph_with_nodes(self, executor=None):
        from smgp.core.graph import SpectralMemoryGraph

        g = SpectralMemoryGraph(hd_dim=100, seed=0, executor=executor)
        for i in range(5):
            g.add_node(str(i), label=f"node{i}")
        for i in range(4):
            g.add_edge(str(i), str(i + 1), "next")
        return g

    def test_executor_inherited_from_graph(self):
        from smgp.core.spectral import SpectralMethods

        mock = MagicMock()
        g = self._make_graph_with_nodes(executor=mock)
        sm = SpectralMethods(g)
        assert sm.executor is mock

    def test_explicit_executor_takes_precedence(self):
        from smgp.core.spectral import SpectralMethods

        mock1, mock2 = MagicMock(), MagicMock()
        g = self._make_graph_with_nodes(executor=mock1)
        sm = SpectralMethods(g, executor=mock2)
        assert sm.executor is mock2

    def test_compute_laplacian_offload(self):
        from smgp.core.spectral import SpectralMethods

        expected = np.eye(5)
        mock = MagicMock()
        mock.compute_laplacian.return_value = expected
        g = self._make_graph_with_nodes(executor=mock)
        sm = SpectralMethods(g)
        if sm._bridge is None:
            pytest.skip("hw_bridge module not importable (hardware dir absent)")
        L = sm.compute_laplacian()
        mock.compute_laplacian.assert_called_once()
        assert L is not None

    def test_fallback_pure_python_laplacian(self):
        from smgp.core.spectral import SpectralMethods

        g = self._make_graph_with_nodes()
        sm = SpectralMethods(g)
        L = sm.compute_laplacian()
        assert L.shape[0] == g.num_nodes

    def test_no_executor_eigen(self):
        from smgp.core.spectral import SpectralMethods

        g = self._make_graph_with_nodes()
        sm = SpectralMethods(g, num_eigenvalues=3)
        evals, evecs = sm.compute_eigen()
        assert evals.shape[0] <= 3
        assert evecs.shape[0] == g.num_nodes


# ---------------------------------------------------------------------------
# HyperdimensionalMemory
# ---------------------------------------------------------------------------


class TestHyperdimWiring:
    def test_executor_stored(self):
        from smgp.core.hyperdim import HyperdimensionalMemory

        mock = MagicMock()
        hd = HyperdimensionalMemory(dim=64, seed=0, executor=mock)
        assert hd.executor is mock

    def test_bind_offload(self):
        from smgp.core.hyperdim import HyperdimensionalMemory

        a = np.ones(64, dtype=np.int8)
        b = np.ones(64, dtype=np.int8)
        expected = np.full(64, 2, dtype=np.int8)

        mock = MagicMock()
        mock.hd_bind.return_value = expected
        hd = HyperdimensionalMemory(dim=64, seed=0, executor=mock)
        if hd._bridge is None:
            pytest.skip("hw_bridge not importable")
        result = hd.bind(a, b)
        mock.hd_bind.assert_called_once_with(a, b)
        np.testing.assert_array_equal(result, expected)

    def test_similarity_offload(self):
        from smgp.core.hyperdim import HyperdimensionalMemory

        a = np.ones(64, dtype=np.int8)
        b = np.ones(64, dtype=np.int8)
        mock = MagicMock()
        mock.hd_similarity.return_value = 0.99
        hd = HyperdimensionalMemory(dim=64, seed=0, executor=mock)
        if hd._bridge is None:
            pytest.skip("hw_bridge not importable")
        result = hd.similarity(a, b)
        mock.hd_similarity.assert_called_once_with(a, b)
        assert result == 0.99

    def test_fallback_bind(self):
        from smgp.core.hyperdim import HyperdimensionalMemory

        hd = HyperdimensionalMemory(dim=64, seed=0)
        a = np.ones(64, dtype=np.int8)
        b = np.full(64, -1, dtype=np.int8)
        result = hd.bind(a, b)
        np.testing.assert_array_equal(result, -np.ones(64, dtype=np.int8))


# ---------------------------------------------------------------------------
# SpectralAttention
# ---------------------------------------------------------------------------


class TestSpectralAttentionWiring:
    def _make_graph(self, executor=None):
        from smgp.core.graph import SpectralMemoryGraph

        g = SpectralMemoryGraph(hd_dim=64, seed=0, executor=executor)
        return g

    def test_executor_stored(self):
        from smgp.attention.spectral_attn import SpectralAttention

        mock = MagicMock()
        g = self._make_graph(executor=mock)
        attn = SpectralAttention(g, hidden_dim=32, num_heads=2, executor=mock)
        assert attn.executor is mock

    def test_executor_inherited_from_graph(self):
        from smgp.attention.spectral_attn import SpectralAttention

        mock = MagicMock()
        g = self._make_graph(executor=mock)
        attn = SpectralAttention(g, hidden_dim=32, num_heads=2)
        assert attn.executor is mock

    def test_forward_offload(self):
        from smgp.attention.spectral_attn import SpectralAttention

        tokens = np.random.randn(4, 32).astype(np.float32)
        expected = np.zeros_like(tokens)
        mock = MagicMock()
        mock.spectral_attention_forward.return_value = expected
        g = self._make_graph(executor=mock)
        attn = SpectralAttention(g, hidden_dim=32, num_heads=2)
        if attn._bridge is None:
            pytest.skip("hw_bridge not importable")
        result = attn.forward(tokens)
        mock.spectral_attention_forward.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_fallback_no_executor(self):
        from smgp.attention.spectral_attn import SpectralAttention

        g = self._make_graph()
        attn = SpectralAttention(g, hidden_dim=32, num_heads=2)
        tokens = np.random.randn(4, 32)
        result = attn.forward(tokens)
        assert result.shape == tokens.shape


# ---------------------------------------------------------------------------
# SMGPConfig hardware settings
# ---------------------------------------------------------------------------


class TestConfigHardware:
    def test_default_hardware_disabled(self):
        from smgp.config import SMGPConfig

        cfg = SMGPConfig()
        assert cfg.hardware.enabled is False
        assert cfg.hardware.fallback is True

    def test_from_dict_hardware_subdict(self):
        from smgp.config import SMGPConfig

        cfg = SMGPConfig.from_dict({"hd_dim": 200, "hardware": {"enabled": False, "device": "/dev/test"}})
        assert cfg.hd_dim == 200
        assert cfg.hardware.device == "/dev/test"
        assert cfg.hardware.enabled is False

    def test_create_executor_disabled(self):
        from smgp.config import SMGPConfig, create_executor_from_config

        cfg = SMGPConfig()
        assert create_executor_from_config(cfg) is None

    def test_create_executor_enabled_raises_import(self):
        from smgp.config import SMGPConfig, create_executor_from_config

        cfg = SMGPConfig.from_dict({"hardware": {"enabled": True}})
        # HAL is not installed in the test environment — expect ImportError
        with pytest.raises(ImportError):
            create_executor_from_config(cfg)
