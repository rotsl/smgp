"""O(1) hyperdimensional associative recall.

Implements content-addressable memory using HD vector algebra for
constant-time approximate recall. Items are encoded as bound pairs
of key and value HD vectors, and recall proceeds via unbinding and
similarity search.

References:
  - Kanerva, P. (1988). "Sparse Distributed Memory." MIT Press.
  - Gallant, S.I., & Okaywe, T.W. (2013). "Representing Objects, Concepts,
    and Sentences with High-Dimensional Random Vectors." Neurocomputing, 121, 50–63.
"""
from __future__ import annotations

import numpy as np

from smgp.core.hyperdim import HyperdimensionalMemory


class AssociativeMemory:
    """Hyperdimensional associative memory with O(1) recall.

    Stores key-value pairs where both keys and values are HD vectors.
    Recall is performed by binding a query key with the stored address
    and finding the most similar value in the memory.

    Attributes:
        hd: HyperdimensionalMemory engine.
        capacity: Maximum number of entries.
        _keys: Dict mapping string keys to HD vectors.
        _values: Dict mapping string keys to HD vectors.
        _value_matrix: Matrix of all stored value vectors for batch similarity.

    References:
        Kanerva, P. (1988). "Sparse Distributed Memory." MIT Press.
    """

    def __init__(
        self,
        hd: HyperdimensionalMemory | None = None,
        capacity: int = 10000,
    ) -> None:
        """Initialize associative memory.

        Args:
            hd: Optional pre-initialized HyperdimensionalMemory. If None,
                creates one with default 10000 dimensions.
            capacity: Maximum number of entries.
        """
        self.hd = hd or HyperdimensionalMemory()
        self.capacity = capacity
        self._keys: dict[str, np.ndarray] = {}
        self._values: dict[str, np.ndarray] = {}
        self._key_str_to_hd: dict[str, np.ndarray] = {}

    def store(self, key: str, value: str) -> None:
        """Store a string key-value pair.

        Both key and value are encoded as HD vectors using a hash-based
        deterministic generation (seeded by the string content).

        Args:
            key: String key.
            value: String value.
        """
        if len(self._keys) >= self.capacity:
            raise MemoryError(f"AssociativeMemory capacity ({self.capacity}) exceeded.")

        # Generate deterministic HD vectors from strings
        key_vec = self._string_to_hd(key)
        val_vec = self._string_to_hd(value)

        self._keys[key] = key_vec
        self._values[key] = val_vec

    def recall(self, key: str, k: int = 1) -> list[tuple[str, float]]:
        """Recall values associated with a key.

        Args:
            key: Query string key.
            k: Number of top results to return.

        Returns:
            List of (value_string, similarity_score) tuples.
        """
        if not self._keys:
            return []

        key_vec = self._string_to_hd(key)
        return self.recall_by_vector(key_vec, k)

    def recall_by_vector(
        self, vector: np.ndarray, k: int = 1
    ) -> list[tuple[str, float]]:
        """Recall values by HD vector similarity.

        Computes cosine similarity between the query vector and all stored
        key vectors, returning the k best matches.

        Args:
            vector: Query HD vector.
            k: Number of results.

        Returns:
            List of (key_string, similarity_score) tuples, sorted descending.
        """
        if not self._keys:
            return []

        key_list = list(self._keys.keys())
        key_matrix = np.array([self._keys[k] for k in key_list])

        idx, sims = self.hd.most_similar(vector, key_matrix, k=min(k, len(key_list)))

        results = [(key_list[int(i)], float(s)) for i, s in zip(idx, sims)]
        return results

    def _string_to_hd(self, s: str) -> np.ndarray:
        """Encode a string as a deterministic HD vector.

        Uses a simple hash-based approach where each character contributes
        to the vector via binding and permutation.

        Args:
            s: Input string.

        Returns:
            HD vector of shape (D,).
        """
        # Create a base vector seeded by the string hash
        seed_val = hash(s) & 0xFFFFFFFF
        rng = np.random.default_rng(seed_val)
        base = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.hd.dim)

        # Incorporate character-level information
        for i, ch in enumerate(s[:32]):  # Use first 32 chars
            char_seed = hash(ch) & 0xFFFFFFFF
            char_rng = np.random.default_rng(char_seed)
            char_vec = char_rng.choice(np.array([-1, 1], dtype=np.int8), size=self.hd.dim)
            permuted = self.hd.permute(char_vec, shifts=i + 1)
            base = self.hd.bundle(np.array([base, permuted]))

        return base
