"""Tests for the TopologicalAnalyzer module."""
import numpy as np
import pytest

from smgp.core.graph import SpectralMemoryGraph
from smgp.core.topology import TopologicalAnalyzer


@pytest.fixture
def graph():
    """Create a test graph with clusters."""
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    # Cluster 1: tightly connected
    for i in range(5):
        g.add_node(f"c1_{i}", label="cluster1")
    for i in range(4):
        g.add_edge(f"c1_{i}", f"c1_{i+1}", "connected")
    g.add_edge("c1_0", "c1_3", "connected")

    # Cluster 2
    for i in range(5):
        g.add_node(f"c2_{i}", label="cluster2")
    for i in range(4):
        g.add_edge(f"c2_{i}", f"c2_{i+1}", "connected")

    # Bridge
    g.add_edge("c1_2", "c2_0", "bridge")
    return g


class TestTopologicalAnalyzer:
    def test_compute_distance_matrix(self, graph):
        analyzer = TopologicalAnalyzer(graph)
        dist = analyzer.compute_distance_matrix()
        assert dist.shape == (10, 10)
        # Diagonal should be 0
        np.testing.assert_array_almost_equal(np.diag(dist), np.zeros(10))
        # Should be symmetric
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_compute_persistence(self, graph):
        analyzer = TopologicalAnalyzer(graph)
        result = analyzer.compute_persistence()
        assert "diagrams" in result
        assert "betti_numbers" in result
        # H0 should exist
        assert len(result["diagrams"]) >= 1

    def test_wasserstein_distance_same(self, graph):
        analyzer = TopologicalAnalyzer(graph)
        diag = analyzer.compute_persistence()
        # Same diagram should have zero distance
        dist = analyzer.wasserstein_distance(diag, diag)
        assert dist == 0.0

    def test_prune_by_persistence(self, graph):
        analyzer = TopologicalAnalyzer(graph)
        original_nodes = graph.num_nodes
        # Prune with a moderate threshold should preserve most nodes
        pruned = analyzer.prune_by_persistence(threshold=0.5)
        assert pruned.num_nodes <= original_nodes
        # Should preserve at least some nodes
        assert pruned.num_nodes > 0

    def test_stability_check(self, graph):
        analyzer = TopologicalAnalyzer(graph)
        # Same graph should be stable
        assert analyzer.stability_check(graph, max_wasserstein=1.0) is True
