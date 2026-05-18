"""Tests for the MemoryStore module."""
import numpy as np
import pytest

from smgp.memory.store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore(hd_dim=100, capacity=100)


class TestMemoryStore:
    def test_write_read(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        value = np.random.choice([-1, 1], size=100).astype(np.int8)
        store.write(key, value)
        assert store.size == 1

        result = store.read(key)
        assert result is not None
        retrieved_value, similarity = result
        assert retrieved_value.shape == (100,)
        assert similarity > 0.5

    def test_read_empty(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        result = store.read(key)
        assert result is None

    def test_update(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        value1 = np.random.choice([-1, 1], size=100).astype(np.int8)
        value2 = np.random.choice([-1, 1], size=100).astype(np.int8)
        store.write(key, value1)
        assert store.update(key, value2) is True
        assert store.size == 1

    def test_update_not_found(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        value = np.random.choice([-1, 1], size=100).astype(np.int8)
        # Different key - should not update
        different_key = np.random.choice([-1, 1], size=100).astype(np.int8)
        store.write(key, value)
        assert store.update(different_key, value) is False

    def test_delete(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        value = np.random.choice([-1, 1], size=100).astype(np.int8)
        store.write(key, value)
        assert store.delete(key) is True
        assert store.size == 0

    def test_delete_not_found(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        assert store.delete(key) is False

    def test_multiple_writes(self, store):
        for i in range(10):
            key = np.random.choice([-1, 1], size=100).astype(np.int8)
            value = np.random.choice([-1, 1], size=100).astype(np.int8)
            store.write(key, value)
        assert store.size == 10

    def test_entries(self, store):
        key = np.random.choice([-1, 1], size=100).astype(np.int8)
        value = np.random.choice([-1, 1], size=100).astype(np.int8)
        store.write(key, value)
        entries = store.entries()
        assert len(entries) == 1
