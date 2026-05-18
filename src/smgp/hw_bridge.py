"""Hardware bridge: thin adapter between SMGP core modules and an HWExecutor.

This module provides a unified interface so that core operations (spectral,
HD, attention) can transparently delegate to any executor object — real
hardware, cycle-accurate simulator, or a mock for testing — without the
core package importing hardware-specific code at module load time.

Usage::

    from smgp.core.graph import SpectralMemoryGraph
    from hardware.sw.smgp_hal.executor import HWExecutor
    from hardware.sw.smgp_hal.hw_session import HWSession

    session = HWSession(device="/dev/smgpu0")
    executor = HWExecutor(session=session)
    graph = SpectralMemoryGraph(hd_dim=10000, seed=42, executor=executor)
    # All heavy operations now offload to FPGA transparently.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass  # executor type is duck-typed; no hard dependency


class HardwareBridge:
    """Thin adapter between SMGP core classes and an executor object.

    Each method attempts to call the identically-named method on the wrapped
    executor.  If the executor does not expose a method, a ``NotImplementedError``
    is raised so the caller can fall back to pure-Python computation.

    The bridge is intentionally duck-typed: it works with the real
    ``HWExecutor``, a cycle-accurate simulator, or any ``MagicMock`` in tests
    — as long as the object exposes the expected method names.

    Args:
        executor: Any object that exposes hardware-accelerated methods.
    """

    def __init__(self, executor: Any) -> None:
        self._exec = executor

    # ------------------------------------------------------------------
    # Spectral operations
    # ------------------------------------------------------------------

    def compute_laplacian(self, adj_matrix: Any, normalized: bool = True) -> Any:
        """Delegate Laplacian computation to the executor.

        Args:
            adj_matrix: Adjacency matrix (scipy sparse or numpy array).
            normalized: Whether to compute the normalized Laplacian.

        Returns:
            Laplacian matrix (format matches executor output).
        """
        fn = getattr(self._exec, "compute_laplacian", None)
        if fn is None:
            raise NotImplementedError("executor has no compute_laplacian method")
        return fn(adj_matrix, normalized)

    def compute_eigen(
        self, laplacian: Any, num_eigenvalues: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delegate eigen-decomposition to the executor.

        Args:
            laplacian: Laplacian matrix.
            num_eigenvalues: Number of smallest eigenpairs to compute.

        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        fn = getattr(self._exec, "compute_eigen", None)
        if fn is None:
            raise NotImplementedError("executor has no compute_eigen method")
        return fn(laplacian, num_eigenvalues)

    def graph_fourier_transform(
        self, signal: np.ndarray, eigenvectors: np.ndarray
    ) -> np.ndarray:
        """Delegate GFT to the executor."""
        fn = getattr(self._exec, "graph_fourier_transform", None)
        if fn is None:
            raise NotImplementedError("executor has no graph_fourier_transform method")
        return fn(signal, eigenvectors)

    def spectral_convolution(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        eigenvectors: np.ndarray,
        eigenvalues: np.ndarray,
    ) -> np.ndarray:
        """Delegate spectral convolution to the executor."""
        fn = getattr(self._exec, "spectral_convolution", None)
        if fn is None:
            raise NotImplementedError("executor has no spectral_convolution method")
        return fn(signal, kernel, eigenvectors, eigenvalues)

    # ------------------------------------------------------------------
    # Hyperdimensional operations
    # ------------------------------------------------------------------

    def hd_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Delegate HD bind (element-wise multiplication) to the executor."""
        fn = getattr(self._exec, "hd_bind", None)
        if fn is None:
            raise NotImplementedError("executor has no hd_bind method")
        return fn(a, b)

    def hd_unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Delegate HD unbind to the executor (same as bind for bipolar)."""
        fn = getattr(self._exec, "hd_unbind", None)
        if fn is not None:
            return fn(a, b)
        # Fall back to hd_bind since unbind == bind for bipolar vectors
        return self.hd_bind(a, b)

    def hd_bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Delegate HD bundle (majority vote) to the executor."""
        fn = getattr(self._exec, "hd_bundle", None)
        if fn is None:
            raise NotImplementedError("executor has no hd_bundle method")
        return fn(vectors)

    def hd_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Delegate HD similarity (cosine) to the executor."""
        fn = getattr(self._exec, "hd_similarity", None)
        if fn is None:
            raise NotImplementedError("executor has no hd_similarity method")
        return fn(a, b)

    def hd_similarity_search(
        self, query: np.ndarray, candidates: Any, k: int = 5
    ) -> list[tuple[int, float]]:
        """Delegate HD similarity search to the executor."""
        fn = getattr(self._exec, "hd_similarity_search", None)
        if fn is None:
            raise NotImplementedError("executor has no hd_similarity_search method")
        return fn(query, candidates, k)

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def spectral_attention_forward(
        self,
        tokens: np.ndarray,
        num_heads: int,
        num_scales: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Delegate spectral attention forward pass to the executor."""
        fn = getattr(self._exec, "spectral_attention_forward", None)
        if fn is None:
            raise NotImplementedError(
                "executor has no spectral_attention_forward method"
            )
        return fn(tokens, num_heads, num_scales, temperature)
