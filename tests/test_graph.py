"""Tests for the SpectralMemoryGraph core module."""
import pytest

from smgp.core.graph import SpectralMemoryGraph


@pytest.fixture
def graph():
    """Create a small test graph."""
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    g.add_node("A", label="entity", properties={"name": "Alice"})
    g.add_node("B", label="entity", properties={"name": "Bob"})
    g.add_node("C", label="concept", properties={"name": "Science"})
    g.add_edge("A", "B", "knows")
    g.add_edge("A", "C", "studies")
    return g


class TestSpectralMemoryGraph:
    def test_add_node(self, graph):
        assert graph.num_nodes == 3
        data = graph.get_node("A")
        assert data is not None
        assert data["label"] == "entity"
        assert data["properties"]["name"] == "Alice"
        assert data["vector"] is not None
        assert data["vector"].shape == (100,)

    def test_add_edge(self, graph):
        assert graph.num_edges == 2

    def test_add_edge_invalid_nodes(self, graph):
        with pytest.raises(KeyError):
            graph.add_edge("A", "NONEXISTENT", "knows")

    def test_remove_node(self, graph):
        graph.remove_node("C")
        assert graph.num_nodes == 2
        assert graph.get_node("C") is None
        assert graph.num_edges == 1  # edge A->C removed

    def test_remove_node_not_found(self, graph):
        with pytest.raises(KeyError):
            graph.remove_node("NONEXISTENT")

    def test_remove_edge(self, graph):
        edges = graph.edges()
        edge_key = edges[0][2]
        graph.remove_edge(edge_key)
        assert graph.num_edges == 1

    def test_remove_edge_not_found(self, graph):
        with pytest.raises(KeyError):
            graph.remove_edge("nonexistent-key")

    def test_get_node(self, graph):
        data = graph.get_node("B")
        assert data["label"] == "entity"
        assert data["vector"].shape == (100,)

    def test_get_node_not_found(self, graph):
        assert graph.get_node("NONEXISTENT") is None

    def test_get_edge(self, graph):
        edges = graph.edges()
        edge_data = graph.get_edge(edges[0][2])
        assert edge_data is not None
        assert edge_data["source"] == "A"
        assert edge_data["relation"] in ("knows", "studies")

    def test_get_edge_not_found(self, graph):
        assert graph.get_edge("nonexistent-key") is None

    def test_query_similar(self, graph):
        query_vec = graph.get_node("A")["vector"]
        results = graph.query_similar(query_vec, k=2)
        assert len(results) == 2
        assert results[0][0] == "A"  # Most similar to itself
        assert results[0][1] == 1.0  # Perfect similarity

    def test_query_similar_with_type_filter(self, graph):
        query_vec = graph.get_node("A")["vector"]
        results = graph.query_similar(query_vec, k=5, node_type="concept")
        assert len(results) == 1
        assert results[0][0] == "C"

    def test_subgraph(self, graph):
        sub = graph.subgraph(["A", "B"])
        assert sub.num_nodes == 2
        assert sub.num_edges == 1
        assert sub.get_node("C") is None

    def test_adjacency_matrix(self, graph):
        adj = graph.adjacency_matrix()
        assert adj.shape == (3, 3)
        assert adj.nnz >= 2  # At least 2 directed edges

    def test_node_vectors(self, graph):
        vecs = graph.node_vectors()
        assert len(vecs) == 3
        assert all(v.shape == (100,) for v in vecs.values())

    def test_neighbors(self, graph):
        neighbors = graph.neighbors("A")
        assert "B" in neighbors
        assert "C" in neighbors

    def test_predecessors(self, graph):
        preds = graph.predecessors("B")
        assert "A" in preds

    def test_node_to_index(self, graph):
        idx_map = graph.node_to_index()
        assert len(idx_map) == 3
        assert set(idx_map.values()) == {0, 1, 2}
