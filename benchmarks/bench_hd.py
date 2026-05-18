"""ASV benchmarks for hyperdimensional computing operations.

Benchmarks the core HD primitives: bind, unbind, similarity,
and bundle operations at various dimensionalities.

References:
    Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation, 1(2).
    Plate, T.A. (2003). "Holographic Reduced Representation." CSLI Publications.
"""
from __future__ import annotations

import numpy as np

from smgp.core.hyperdim import HyperdimensionalMemory


class HyperdimensionalBenchmarks:
    """Benchmarks for hyperdimensional vector operations."""

    params = [1000, 5000, 10000, 50000]
    param_names = ["dim"]

    def setup(self, dim):
        """Create HD memory engine with given dimensionality."""
        self.hd = HyperdimensionalMemory(dim=dim, seed=42)
        self.vec_a = self.hd.generate(1)[0]
        self.vec_b = self.hd.generate(1)[0]

    def teardown(self, dim):
        """Clean up HD memory engine."""
        del self.hd
        del self.vec_a
        del self.vec_b

    def time_hd_bind_unbind(self, dim):
        """Benchmark binding and unbinding two HD vectors.

        Bind pairs two vectors via element-wise multiplication;
        unbind is self-inverse for bipolar vectors.
        """
        bound = self.hd.bind(self.vec_a, self.vec_b)
        recovered = self.hd.unbind(bound, self.vec_b)

    def time_hd_similarity(self, dim):
        """Benchmark cosine similarity computation between two HD vectors.

        Computes the cosine similarity: (a . b) / (||a|| * ||b||).
        """
        self.hd.similarity(self.vec_a, self.vec_b)

    def time_hd_bundle(self, n, dim):
        """Benchmark bundling (superposing) n HD vectors via element-wise majority.

        Uses n=100 vectors of the specified dimensionality.
        The bundle operation sums vectors element-wise and takes the sign.
        """
        vectors = self.hd.generate(100)
        self.hd.bundle(vectors)


class HyperdimensionalBundleBenchmarks:
    """Parametric benchmarks for bundle operation varying both n and dim."""

    params = ([10, 50, 100, 500, 1000], [1000, 10000])
    param_names = ["n", "dim"]

    def setup(self, n, dim):
        """Create HD memory engine and generate n vectors."""
        self.hd = HyperdimensionalMemory(dim=dim, seed=42)

    def teardown(self, n, dim):
        """Clean up."""
        del self.hd

    def time_hd_bundle(self, n, dim):
        """Bundle n vectors of given dimensionality."""
        vectors = self.hd.generate(n)
        self.hd.bundle(vectors)
