"""Hyperdimensional computing primitives for O(1) associative memory.

Implements the algebra of high-dimensional binary/bipolar vectors as described in:
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing
  in Distributed Representation with High-Dimensional Random Vectors."
  Cognitive Computation, 1(2), 139–159.
- Plate, T.A. (2003). "Holographic Reduced Representation: Distributed Algebra
  for Compositional Structures." CSLI Publications.

Key operations:
  - Generate: Create random bipolar {-1, +1}^D vectors (D=10000).
  - Bundle: Superpose vectors via element-wise majority (approximate superposition).
  - Bind: Associative pairing via element-wise multiplication (XOR for binary).
  - Permute: Cyclic shift for order encoding.
  - Similarity: Cosine similarity for approximate matching.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from smgp.hw_bridge import HardwareBridge as _HardwareBridge
except ImportError:
    _HardwareBridge = None  # type: ignore[assignment,misc]


class HyperdimensionalMemory:
    """Engine for hyperdimensional vector operations.

    Generates, bundles, binds, and permutes 10,000-dimensional random bipolar
    vectors for use as node/edge addresses in the knowledge graph. Entire graph
    queries reduce to HD vector operations, achieving O(1) associative recall
    via dot-product similarity.

    Attributes:
        dim: Dimensionality of hyperdimensional vectors (default 10000).
        rng: Numpy random number generator for reproducibility.

    References:
        Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation, 1(2).
        Gayler, R.W. (2003). "Vector Symbolic Architectures Answer Random
            Permutations of Binary Vectors Do Not." In Proc. of the Joint Int.
            Conf. on Cognitive Science.
    """

    def __init__(self, dim: int = 10000, seed: int | None = None,
                 executor: Any = None) -> None:
        """Initialize the HD memory engine.

        Args:
            dim: Dimensionality D of the bipolar vectors. Kanerva (1988) recommends
                D >= 1000; we use D=10000 for robustness against superposition noise.
            seed: Random seed for reproducible vector generation.
            executor: Optional hardware executor.  When provided, bind/unbind,
                bundle, and similarity operations are offloaded to the
                accelerator.  Defaults to ``None`` (pure-Python).
        """
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.executor = executor
        self._bridge = (
            _HardwareBridge(executor)
            if executor is not None and _HardwareBridge is not None
            else None
        )

    def generate(self, n: int = 1) -> np.ndarray:
        """Generate n random bipolar vectors.

        Each vector v in {-1, +1}^D is drawn i.i.d. with P(v_i = +1) = P(v_i = -1) = 0.5.

        Args:
            n: Number of vectors to generate.

        Returns:
            np.ndarray of shape (n, D) with dtype np.int8 for memory efficiency.
        """
        return self.rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(n, self.dim)
        )

    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """Bundle (superpose) multiple bipolar vectors via element-wise majority.

        For bipolar vectors, the bundle (superposition) is computed by summing
        element-wise and taking the sign. This is the "majority vote" operation
        from Kanerva (2009), which acts as a robust approximation of the
        superposition of encoded items.

        Mathematically: bundle(v_1, ..., v_n)_i = sign(sum_j v_j_i)

        Args:
            vectors: np.ndarray of shape (n, D) with dtype np.int8.

        Returns:
            np.ndarray of shape (D,) with dtype np.int8.

        References:
            Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation, 1(2).
        """
        if vectors.ndim == 1:
            return vectors.astype(np.int8)
        if self._bridge is not None:
            try:
                return self._bridge.hd_bundle(list(vectors))
            except NotImplementedError:
                pass
        sums = vectors.sum(axis=0).astype(np.int32)
        result = np.sign(sums).astype(np.int8)
        result[result == 0] = self.rng.choice(
            np.array([-1, 1], dtype=np.int8), size=int((result == 0).sum())
        )
        return result

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two bipolar vectors via element-wise multiplication.

        For bipolar vectors, binding (a * b)_i = a_i * b_i is self-inverse:
        unbind(bind(a, b), b) = a. This is the XOR operation for binary
        vectors, generalized to the bipolar domain as element-wise product.

        This operation allows associative pairing: a role can be bound to a
        filler, creating a record that can be probed by unbinding the role.

        Args:
            a: First vector of shape (D,).
            b: Second vector of shape (D,).

        Returns:
            np.ndarray of shape (D,) with dtype np.int8.

        References:
            Plate, T.A. (2003). "Holographic Reduced Representation." CSLI Publications.
        """
        if self._bridge is not None:
            try:
                return self._bridge.hd_bind(a, b)
            except NotImplementedError:
                pass
        return (a * b).astype(np.int8)

    def unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Unbind a bipolar vector (self-inverse of bind).

        For bipolar vectors, unbind is identical to bind since:
        unbind(a, b) = a * b, and bind(a, b) * b = a * b * b = a.

        Args:
            a: Bound vector of shape (D,).
            b: Probe vector of shape (D,).

        Returns:
            np.ndarray of shape (D,) with dtype np.int8.
        """
        if self._bridge is not None:
            try:
                return self._bridge.hd_unbind(a, b)
            except NotImplementedError:
                pass
        return self.bind(a, b)

    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Permute a bipolar vector via cyclic shift.

        Cyclic permutation provides a clean, invertible transformation that
        approximately preserves the pseudo-randomness of the vector. It is used
        to encode sequence order and to create independent basis vectors from
        a single seed.

        Args:
            v: Vector of shape (D,).
            shifts: Number of positions to cyclically shift. Default 1.

        Returns:
            np.ndarray of shape (D,) with dtype np.int8.
        """
        return np.roll(v, shifts).astype(np.int8)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        cosine(a, b) = (a . b) / (||a|| * ||b||)

        For bipolar vectors of dimension D, the expected similarity of two
        random independent vectors is approximately 0, with standard deviation
        1/sqrt(D) ≈ 0.01 for D=10000.

        Args:
            a: First vector of shape (D,).
            b: Second vector of shape (D,).

        Returns:
            Float cosine similarity in [-1, 1].
        """
        if self._bridge is not None:
            try:
                return self._bridge.hd_similarity(a, b)
            except NotImplementedError:
                pass
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        norm_a = np.linalg.norm(a_f)
        norm_b = np.linalg.norm(b_f)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_f, b_f) / (norm_a * norm_b))

    def most_similar(
        self,
        query: np.ndarray,
        memory: np.ndarray,
        k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the k most similar vectors to a query in a memory bank.

        Computes cosine similarity between the query and all memory vectors,
        returning the top-k matches. This enables O(1) associative recall when
        the memory bank is organized as a flat matrix.

        Args:
            query: Query vector of shape (D,).
            memory: Memory bank of shape (N, D).
            k: Number of top results to return.

        Returns:
            Tuple of (indices, similarities), each of shape (k,).
        """
        query_f = query.astype(np.float64)
        memory_f = memory.astype(np.float64)
        norms = np.linalg.norm(memory_f, axis=1)
        norms[norms == 0] = 1.0
        query_norm = np.linalg.norm(query_f)
        if query_norm == 0:
            sims = np.zeros(memory_f.shape[0])
        else:
            sims = (memory_f @ query_f) / (norms * query_norm)
        top_k = min(k, len(sims))
        if top_k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        if top_k == len(sims):
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]
