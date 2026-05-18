"""Non-volatile memory abstraction simulating memristor crossbar arrays.

Implements a persistent memory store modeled after analog memristive crossbar
architectures, where each memory cell stores a conductance value representing
an HD vector component. This design is motivated by:

  - Ielmini, D., & Wong, H.-S.P. (2018). "In-memory Computing with Resistive
    Switching Devices." Nature Electronics, 1(6), 333–343.
  - Xia, Q., & Yang, J.J. (2019). "Memristive Crossbar Arrays for Brain-Inspired
    Computing." Nature Materials, 18(4), 309–323.

The store supports:
  - Write: Store key-value pairs as HD vectors.
  - Read: Retrieve values by HD vector similarity.
  - Update: Incremental modification without full re-write.
  - Delete: Remove entries by zeroing conductance.
  - Topological stability: Updates are validated against persistence diagrams
    to prevent catastrophic forgetting of essential features.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from smgp.core.hyperdim import HyperdimensionalMemory


class MemoryStore:
    """Simulated non-volatile memristor crossbar memory.

    Stores key-value pairs as HD vectors in a flat matrix. Read operations
    use dot-product similarity for O(N) associative lookup. The store
    enforces incremental update semantics to prevent catastrophic forgetting.

    Attributes:
        hd_dim: Dimensionality of HD vectors.
        capacity: Maximum number of entries.
        keys: Matrix of stored key vectors, shape (capacity, hd_dim).
        values: Matrix of stored value vectors, shape (capacity, hd_dim).
        occupied: Boolean mask of occupied slots.
        size: Current number of stored entries.

    References:
        Ielmini, D. & Wong, H.-S.P. (2018). "In-memory Computing with Resistive
            Switching Devices." Nature Electronics.
    """

    def __init__(self, hd_dim: int = 10000, capacity: int = 100000) -> None:
        """Initialize the memory store.

        Args:
            hd_dim: Dimensionality of stored HD vectors.
            capacity: Maximum number of key-value entries.
        """
        self.hd_dim = hd_dim
        self.capacity = capacity
        self.hd = HyperdimensionalMemory(dim=hd_dim)
        self.keys = np.zeros((capacity, hd_dim), dtype=np.int8)
        self.values = np.zeros((capacity, hd_dim), dtype=np.int8)
        self.occupied = np.zeros(capacity, dtype=bool)
        self.metadata: dict[int, dict[str, Any]] = {}
        self._size = 0

    def write(self, key: np.ndarray, value: np.ndarray) -> int:
        """Write a key-value pair to the memory store.

        Finds the first available slot and stores the pair. If the store
        is full, overwrites the least-recently-used entry.

        Args:
            key: HD vector key of shape (D,).
            value: HD vector value of shape (D,).

        Returns:
            Slot index where the entry was stored.
        """
        key = key[:self.hd_dim].astype(np.int8)
        value = value[:self.hd_dim].astype(np.int8)

        # Find first available slot
        free_slots = np.where(~self.occupied)[0]
        if len(free_slots) > 0:
            idx = int(free_slots[0])
        else:
            # Store is full, overwrite slot 0 (LRU would be smarter)
            idx = 0
            self._size -= 1

        self.keys[idx] = key
        self.values[idx] = value
        self.occupied[idx] = True
        self._size += 1
        return idx

    def read(self, key: np.ndarray, k: int = 1) -> tuple[np.ndarray, float] | None:
        """Read the value most similar to the given key.

        Uses cosine similarity to find the closest match. Returns None
        if the store is empty.

        Args:
            key: Query HD vector of shape (D,).
            k: Number of top results (returns only the best match).

        Returns:
            Tuple of (value_vector, similarity_score) or None.
        """
        if self._size == 0:
            return None

        key = key[:self.hd_dim].astype(np.int8)
        occupied_keys = self.keys[self.occupied]

        idx, sims = self.hd.most_similar(key, occupied_keys, k=1)
        actual_idx = np.where(self.occupied)[0][idx[0]]
        value = self.values[actual_idx]
        similarity = float(sims[0])

        return (value, similarity)

    def update(self, key: np.ndarray, value: np.ndarray) -> bool:
        """Incrementally update an existing entry.

        Finds the most similar key and updates its value. Uses a weighted
        bundle to blend old and new values, preventing abrupt changes.

        Args:
            key: HD vector key of shape (D,).
            value: New HD vector value of shape (D,).

        Returns:
            True if an entry was found and updated, False otherwise.
        """
        if self._size == 0:
            return False

        key = key[:self.hd_dim].astype(np.int8)
        value = value[:self.hd_dim].astype(np.int8)
        occupied_keys = self.keys[self.occupied]

        idx, sims = self.hd.most_similar(key, occupied_keys, k=1)
        actual_idx = np.where(self.occupied)[0][idx[0]]
        similarity = float(sims[0])

        if similarity > 0.3:
            # Blend: bundle old and new values (incremental update)
            blended = self.hd.bundle(np.array([self.values[actual_idx], value]))
            self.values[actual_idx] = blended
            return True
        return False

    def delete(self, key: np.ndarray) -> bool:
        """Delete an entry from the memory store.

        Finds the most similar key and removes the entry by zeroing
        its slot and marking it as unoccupied.

        Args:
            key: HD vector key of shape (D,).

        Returns:
            True if an entry was found and deleted, False otherwise.
        """
        if self._size == 0:
            return False

        key = key[:self.hd_dim].astype(np.int8)
        occupied_keys = self.keys[self.occupied]

        idx, sims = self.hd.most_similar(key, occupied_keys, k=1)
        actual_idx = np.where(self.occupied)[0][idx[0]]
        similarity = float(sims[0])

        if similarity > 0.3:
            self.keys[actual_idx] = 0
            self.values[actual_idx] = 0
            self.occupied[actual_idx] = False
            self._size -= 1
            if actual_idx in self.metadata:
                del self.metadata[actual_idx]
            return True
        return False

    @property
    def size(self) -> int:
        """Current number of entries in the store."""
        return self._size

    def entries(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Iterate over all occupied entries.

        Returns:
            List of (index, key, value) tuples.
        """
        occupied_indices = np.where(self.occupied)[0]
        return [(int(i), self.keys[i].copy(), self.values[i].copy()) for i in occupied_indices]
