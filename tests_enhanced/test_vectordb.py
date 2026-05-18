"""Tests for Vector Database Bridge integration."""
from __future__ import annotations

import pytest

HAS_QDRANT = False
try:
    import qdrant_client  # noqa: F401
    HAS_QDRANT = True
except ImportError:
    pass


class TestVectorDBSyncerImport:
    """Test that VectorDBSyncer can be imported regardless of deps."""

    def test_import_module(self):
        """Module should be importable."""
        from smgp.integration.vectordb import VectorDBSyncer
        assert VectorDBSyncer is not None

    def test_unsupported_backend_raises(self):
        """Unsupported backend should raise ValueError."""
        from smgp.core.graph import SpectralMemoryGraph
        from smgp.integration.vectordb import VectorDBSyncer
        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        with pytest.raises(ValueError, match="Unsupported backend"):
            VectorDBSyncer(graph, backend="unknown_db")

    def test_qdrant_import_error_without_client(self):
        """Attempting to sync without qdrant-client should raise ImportError."""
        from smgp.core.graph import SpectralMemoryGraph
        from smgp.integration.vectordb import VectorDBSyncer

        if HAS_QDRANT:
            pytest.skip("qdrant-client is installed, cannot test import error")

        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        graph.add_node("n1", label="test")
        syncer = VectorDBSyncer(graph, backend="qdrant")
        with pytest.raises((ImportError, ModuleNotFoundError), match="qdrant"):
            syncer.sync_nodes()


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client is not installed")
class TestVectorDBSyncerQdrant:
    """Tests for Qdrant backend with in-memory client."""

    def test_sync_empty_graph(self):
        """Syncing an empty graph should produce zero results."""
        from smgp.core.graph import SpectralMemoryGraph
        from smgp.integration.vectordb import VectorDBSyncer

        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        syncer = VectorDBSyncer(graph, backend="qdrant")
        result = syncer.sync_nodes("test_empty")
        assert result["num_synced"] == 0

    def test_sync_and_query(self):
        """Sync nodes, then query and verify results."""
        from smgp.core.graph import SpectralMemoryGraph
        from smgp.integration.vectordb import VectorDBSyncer

        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        graph.add_node("node_1", label="animal", properties={"name": "cat"})
        graph.add_node("node_2", label="animal", properties={"name": "dog"})
        graph.add_node("node_3", label="concept", properties={"name": "quantum"})

        syncer = VectorDBSyncer(graph, backend="qdrant")
        result = syncer.sync_nodes("test_sync")
        assert result["num_synced"] == 3

        # Query using a known node's vector
        node_data = graph.get_node("node_1")
        assert node_data is not None
        query_vec = node_data["vector"]

        results = syncer.query_external(query_vec, "test_sync", k=3)
        assert len(results) >= 1
        node_ids = [r["node_id"] for r in results]
        assert "node_1" in node_ids

    def test_hybrid_search(self):
        """Hybrid search should filter by text."""
        from smgp.core.graph import SpectralMemoryGraph
        from smgp.integration.vectordb import VectorDBSyncer

        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        graph.add_node("cat", label="animal")
        graph.add_node("dog", label="animal")
        graph.add_node("quantum", label="concept")

        syncer = VectorDBSyncer(graph, backend="qdrant")
        syncer.sync_nodes("test_hybrid")

        node_data = graph.get_node("cat")
        assert node_data is not None
        query_vec = node_data["vector"]

        results = syncer.hybrid_search(
            query_vec,
            text_filter="animal",
            collection_name="test_hybrid",
            k=5,
        )
        # All returned results should contain "animal" in label
        for r in results:
            assert "animal" in r.get("label", "").lower()

    def test_query_nonexistent_collection(self):
        """Querying a nonexistent collection should raise or return empty."""
        from smgp.core.graph import SpectralMemoryGraph
        from smgp.integration.vectordb import VectorDBSyncer

        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        graph.add_node("x", label="test")
        syncer = VectorDBSyncer(graph, backend="qdrant")

        query_vec = graph.get_node("x")["vector"]
        with pytest.raises(Exception):
            syncer.query_external(query_vec, "nonexistent_collection", k=5)
