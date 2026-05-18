"""Tests for pluggable pruning policies.

Each test builds a small graph, records accesses via the policy's
:meth:`touch` method, and verifies that the correct nodes are selected
for removal.
"""
from __future__ import annotations

import time

from smgp.core.graph import SpectralMemoryGraph
from smgp.memory.enhanced_lifecycle import EnhancedMemoryLifecycle
from smgp.memory.prune_policies import (
    AccessCountPrunePolicy,
    LRUPrunePolicy,
    TimeBasedPrunePolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n: int = 10) -> SpectralMemoryGraph:
    """Create a graph with *n* nodes and sequential edges."""
    g = SpectralMemoryGraph(hd_dim=100, seed=0)
    for i in range(n):
        g.add_node(f"n{i}", label=f"node_{i}")
    for i in range(n - 1):
        g.add_edge(f"n{i}", f"n{i + 1}", "next")
    return g


# ---------------------------------------------------------------------------
# LRUPrunePolicy
# ---------------------------------------------------------------------------

class TestLRUPrunePolicy:
    """Tests for :class:`LRUPrunePolicy`."""

    def test_prune_oldest_nodes(self) -> None:
        g = _make_graph(10)
        policy = LRUPrunePolicy()

        # Touch nodes 0..8; node 9 is never touched → oldest
        for i in range(9):
            policy.touch(f"n{i}")

        # Small sleep so ordering is deterministic
        time.sleep(0.01)
        policy.touch("n0")  # refresh node 0

        pruned_ids = policy.select_nodes_to_prune(g, count=3)
        assert len(pruned_ids) == 3
        # node 9 was never touched → should be first (epoch 0)
        assert "n9" in pruned_ids
        # node 0 was refreshed, so nodes 1-8 should be pruned before it
        assert "n0" not in pruned_ids or pruned_ids.index("n0") > pruned_ids.index("n1")

    def test_prune_with_enhanced_lifecycle(self) -> None:
        g = _make_graph(10)
        policy = LRUPrunePolicy()
        for i in range(10):
            policy.touch(f"n{i}")
        time.sleep(0.01)
        policy.touch("n9")  # refresh the last node

        lc = EnhancedMemoryLifecycle(g, policy=policy)
        result = lc.prune(count=3)
        assert result.num_nodes == 7
        # n9 was refreshed so it should survive
        assert "n9" in result.nodes()

    def test_empty_graph(self) -> None:
        g = SpectralMemoryGraph(hd_dim=100, seed=0)
        policy = LRUPrunePolicy()
        assert policy.select_nodes_to_prune(g, count=5) == []


# ---------------------------------------------------------------------------
# TimeBasedPrunePolicy
# ---------------------------------------------------------------------------

class TestTimeBasedPrunePolicy:
    """Tests for :class:`TimeBasedPrunePolicy`."""

    def test_no_expired_nodes(self) -> None:
        g = _make_graph(5)
        policy = TimeBasedPrunePolicy(ttl_seconds=3600.0)
        for i in range(5):
            policy.touch(f"n{i}")
        # All nodes touched just now → none expired
        assert policy.select_nodes_to_prune(g, count=3) == []

    def test_expired_nodes_pruned(self) -> None:
        g = _make_graph(5)
        policy = TimeBasedPrunePolicy(ttl_seconds=0.0)  # instant expiry
        # Don't touch any nodes — they all have epoch-0 access times
        expired = policy.select_nodes_to_prune(g, count=3)
        assert len(expired) == 3
        for nid in expired:
            assert nid in g.nodes()

    def test_partial_expiry(self) -> None:
        g = _make_graph(5)
        policy = TimeBasedPrunePolicy(ttl_seconds=0.05)
        # Touch nodes 0,1 just now — they are fresh
        policy.touch("n0")
        policy.touch("n1")
        # Nodes 2,3,4 were never touched (epoch 0) → expired
        expired = policy.select_nodes_to_prune(g, count=10)
        assert set(expired) == {"n2", "n3", "n4"}


# ---------------------------------------------------------------------------
# AccessCountPrunePolicy
# ---------------------------------------------------------------------------

class TestAccessCountPrunePolicy:
    """Tests for :class:`AccessCountPrunePolicy`."""

    def test_below_threshold(self) -> None:
        g = _make_graph(10)
        policy = AccessCountPrunePolicy(min_accesses=2)

        # Touch nodes 0-4 once, 5-9 three times
        for i in range(5):
            policy.touch(f"n{i}")
        for i in range(5, 10):
            policy.touch(f"n{i}")
            policy.touch(f"n{i}")
            policy.touch(f"n{i}")

        pruned = policy.select_nodes_to_prune(g, count=10)
        assert set(pruned) == {f"n{i}" for i in range(5)}

    def test_count_limit(self) -> None:
        g = _make_graph(10)
        policy = AccessCountPrunePolicy(min_accesses=5)
        # No node accessed enough → all eligible
        pruned = policy.select_nodes_to_prune(g, count=3)
        assert len(pruned) == 3

    def test_with_enhanced_lifecycle(self) -> None:
        g = _make_graph(6)
        policy = AccessCountPrunePolicy(min_accesses=2)
        for i in range(3):
            policy.touch(f"n{i}")
            policy.touch(f"n{i}")
        # nodes 3-5 only touched once

        lc = EnhancedMemoryLifecycle(g, policy=policy)
        result = lc.prune(count=10)
        assert result.num_nodes == 3
        for i in range(3):
            assert f"n{i}" in result.nodes()
