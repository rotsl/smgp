"""ASV benchmarks for spectral graph analysis operations.

Benchmarks Laplacian computation, eigen-decomposition, and
Chebyshev polynomial convolution on graphs of varying size.

References:
    Chung, F.R.K. (1997). "Spectral Graph Theory." CBMS No. 92.
    Defferrard, M., et al. (2016). "ChebNet." NeurIPS.
"""
from __future__ import annotations

import numpy as np

from smgp.core.graph import SpectralMemoryGraph
from smgp.core.spectral import SpectralMethods


class SpectralBenchmarks:
    """Benchmarks for spectral graph analysis methods."""

    params = [50, 100, 250, 500]
    param_names = ["n"]

    def setup(self, n):
        """Create a graph and spectral methods instance."""
        self.graph = SpectralMemoryGraph(hd_dim=10000, seed=42)
        # Build a connected graph with chain + some random edges
        for i in range(n):
            self.graph.add_node(f"node_{i}", label="spectral_bench")
        for i in range(n - 1):
            self.graph.add_edge(f"node_{i}", f"node_{i + 1}", "chain")
        # Add skip edges for richer spectral structure
        rng = np.random.default_rng(42)
        for _ in range(n // 2):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                self.graph.add_edge(f"node_{i}", f"node_{j}", "skip")
        self.spectral = SpectralMethods(self.graph, num_eigenvalues=min(32, n - 1))

    def teardown(self, n):
        """Clean up graph and spectral instances."""
        del self.graph
        del self.spectral

    def time_laplacian(self, n):
        """Benchmark computing the normalized graph Laplacian.

        Measures construction of L_norm = I - D^{-1/2} A D^{-1/2}
        including adjacency matrix extraction and degree computation.
        """
        # Reset cache to force recomputation
        self.spectral._laplacian = None
        self.spectral.compute_laplacian()

    def time_eigen_decomposition(self, n):
        """Benchmark eigen-decomposition of the graph Laplacian.

        Uses scipy.sparse.linalg.eigsh to compute the smallest
        eigenvalues and corresponding eigenvectors.
        """
        # Reset caches to force recomputation
        self.spectral._laplacian = None
        self.spectral._eigenvalues = None
        self.spectral._eigenvectors = None
        self.spectral.compute_eigen()

    def time_chebyshev_convolution(self, n):
        """Benchmark Chebyshev polynomial spectral convolution.

        Applies a 5-th order Chebyshev filter to a random signal
        defined on the graph nodes.
        """
        # Reset caches to force recomputation
        self.spectral._laplacian = None
        self.spectral._eigenvalues = None
        self.spectral._eigenvectors = None
        signal = np.random.default_rng(42).standard_normal(n)
        coefficients = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        self.spectral.chebyshev_convolution(signal, coefficients)
