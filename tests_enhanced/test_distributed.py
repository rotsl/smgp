"""Tests for Distributed Memory Graph (Sharding)."""
from __future__ import annotations

import numpy as np
import pytest


class TestDistributedGraphBackend:
    """Tests for the distributed graph with sharding."""

    def test_import(self):
        """Module should be importable."""
        from smgp.core.distributed import DistributedGraphBackend
        assert DistributedGraphBackend is not None

    def test_initialization(self):
        """Should initialize with correct number of shards."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=4, hd_dim=1000, seed=42)
        assert dg.num_shards == 4
        assert len(dg.shards) == 4
        assert dg.num_nodes == 0

    def test_single_shard(self):
        """Should work with num_shards=1."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=1, hd_dim=1000)
        dg.add_node("n1", label="test")
        assert dg.num_nodes == 1
        node = dg.get_node("n1")
        assert node is not None
        assert node["label"] == "test"

    def test_invalid_num_shards(self):
        """num_shards < 1 should raise ValueError."""
        from smgp.core.distributed import DistributedGraphBackend
        with pytest.raises(ValueError, match="num_shards must be >= 1"):
            DistributedGraphBackend(num_shards=0)

    def test_add_node_routes_to_shard(self):
        """Nodes should be routed to shards via consistent hashing."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000, seed=42)

        # Deterministic shard assignment based on node_id
        shard_0 = dg._get_shard("node_0")
        shard_1 = dg._get_shard("node_1")

        dg.add_node("node_0", label="a")
        dg.add_node("node_1", label="b")

        assert dg.get_node("node_0") is not None
        assert dg.get_node("node_1") is not None
        # At least one node should be on a specific shard
        assert dg.shards[shard_0].num_nodes >= 1
        assert dg.shards[shard_1].num_nodes >= 1

    def test_node_distribution_across_shards(self):
        """With 20 nodes and 2 shards, not all nodes should be on one shard."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000, seed=42)

        for i in range(20):
            dg.add_node(f"node_{i}", label=f"type_{i % 3}")

        dist = dg.shard_distribution()
        # Not all nodes on a single shard
        assert 0 < dist[0] < 20
        assert 0 < dist[1] < 20
        # Total nodes via property
        assert dg.num_nodes == 20

    def test_get_node_missing(self):
        """get_node for non-existent node should return None."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000)
        assert dg.get_node("nonexistent") is None

    def test_add_edge_same_shard(self):
        """Edge between nodes on same shard should work."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000, seed=42)

        # Find two node IDs that hash to same shard
        shard_idx = dg._get_shard("same_a")
        # Keep generating IDs until we find one on same shard
        counter = 0
        other_id = None
        for i in range(100):
            candidate = f"same_b_{i}"
            if dg._get_shard(candidate) == shard_idx:
                other_id = candidate
                break
            counter += 1
        assert other_id is not None, "Could not find node on same shard"

        dg.add_node("same_a", label="x")
        dg.add_node(other_id, label="y")
        edge_key = dg.add_edge("same_a", other_id, "related")
        assert edge_key is not None
        assert dg.num_edges >= 1

    def test_add_edge_cross_shard(self):
        """Edge between nodes on different shards should be stored on both."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000, seed=42)

        # Find node IDs on different shards
        shard_0_node = None
        shard_1_node = None
        for i in range(100):
            nid = f"cross_{i}"
            s = dg._get_shard(nid)
            if s == 0 and shard_0_node is None:
                shard_0_node = nid
            elif s == 1 and shard_1_node is None:
                shard_1_node = nid
            if shard_0_node and shard_1_node:
                break

        assert shard_0_node is not None and shard_1_node is not None

        dg.add_node(shard_0_node, label="a")
        dg.add_node(shard_1_node, label="b")
        edge_key = dg.add_edge(shard_0_node, shard_1_node, "cross")

        assert edge_key is not None
        # Edge should exist on source shard
        assert dg.shards[0].num_edges >= 1
        # Edge should also exist on target shard
        assert dg.shards[1].num_edges >= 1

    def test_add_edge_missing_node_raises(self):
        """Adding edge with missing source or target should raise KeyError."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000)
        dg.add_node("exists", label="x")
        with pytest.raises(KeyError, match="Target node"):
            dg.add_edge("exists", "missing", "relation")

    def test_remove_node(self):
        """Removing a node should work and update counts."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000, seed=42)
        dg.add_node("to_remove", label="temp")
        assert dg.num_nodes == 1
        dg.remove_node("to_remove")
        assert dg.num_nodes == 0
        assert dg.get_node("to_remove") is None

    def test_remove_node_missing_raises(self):
        """Removing a non-existent node should raise KeyError."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000)
        with pytest.raises(KeyError, match="not found"):
            dg.remove_node("nonexistent")

    def test_query_similar(self):
        """query_similar should search all shards and return merged results."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=10000, seed=42)

        # Add nodes distributed across shards
        for i in range(20):
            dg.add_node(f"qnode_{i}", label=f"type_{i % 3}")

        # Query with a vector similar to one of the nodes
        node_data = dg.get_node("qnode_5")
        assert node_data is not None
        query_vec = node_data["vector"]

        results = dg.query_similar(query_vec, k=3)
        assert len(results) <= 3
        assert len(results) >= 1

        # Results should be sorted by descending similarity
        for i in range(1, len(results)):
            assert results[i][1] <= results[i - 1][1]

        # Top result should be the same node (similarity ~= 1.0)
        top_node_id, top_score = results[0]
        assert top_node_id == "qnode_5"
        assert top_score > 0.99

    def test_query_similar_empty(self):
        """query_similar on empty graph should return empty list."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000)
        vec = np.random.choice([-1, 1], size=1000).astype(np.int8)
        results = dg.query_similar(vec, k=5)
        assert results == []

    def test_nodes_returns_unique(self):
        """nodes() should return unique node IDs."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=2, hd_dim=1000, seed=42)
        for i in range(10):
            dg.add_node(f"unique_{i}", label="test")
        all_nodes = dg.nodes()
        # May have duplicates from cross-shard edge replication,
        # but the method should deduplicate
        assert len(all_nodes) == len(set(all_nodes))

    def test_repr(self):
        """repr should be informative."""
        from smgp.core.distributed import DistributedGraphBackend
        dg = DistributedGraphBackend(num_shards=4, hd_dim=5000)
        r = repr(dg)
        assert "DistributedGraphBackend" in r
        assert "num_shards=4" in r
