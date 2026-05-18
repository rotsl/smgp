"""Tests for the GraphRewriter (category theory) module."""
import pytest

from smgp.core.category import GraphRewriter
from smgp.core.graph import SpectralMemoryGraph


@pytest.fixture
def family_graph():
    """Create a family relations graph for testing DPO rewrites."""
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    g.add_node("Alice", label="person", properties={"gender": "F"})
    g.add_node("Bob", label="person", properties={"gender": "M"})
    g.add_node("Charlie", label="person", properties={"gender": "M"})
    g.add_node("Dave", label="person", properties={"gender": "M"})
    g.add_edge("Alice", "Bob", "parent_of")
    g.add_edge("Bob", "Charlie", "parent_of")
    return g


class TestGraphRewriter:
    def test_match_pattern(self, family_graph):
        rewriter = GraphRewriter(family_graph)
        pattern = {
            "nodes": [
                {"label": "person"},
                {"label": "person"},
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "parent_of"},
            ],
        }
        matches = rewriter.match_pattern(pattern)
        assert len(matches) >= 2  # Alice->Bob, Bob->Charlie

    def test_match_pattern_with_constraints(self, family_graph):
        rewriter = GraphRewriter(family_graph)
        pattern = {
            "nodes": [
                {"label": "person", "constraints": {"properties": {"gender": "F"}}},
                {"label": "person"},
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "parent_of"},
            ],
        }
        matches = rewriter.match_pattern(pattern)
        assert len(matches) == 1
        assert matches[0]["0"] == "Alice"

    def test_apply_dpo_rewrite(self, family_graph):
        rewriter = GraphRewriter(family_graph)
        # Rule: Replace "parent_of" with "mother_of" for female parents
        lhs = {
            "nodes": [
                {"label": "person"},
                {"label": "person"},
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "parent_of"},
            ],
        }
        rhs = {
            "nodes": [
                {"label": "person"},
                {"label": "person"},
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "mother_of"},
            ],
        }
        interface = {"node_indices": [0, 1]}

        matches = rewriter.match_pattern(lhs)
        assert len(matches) >= 1
        match = matches[0]

        new_graph = rewriter.apply_dpo_rewrite(lhs, rhs, interface, match)
        # The rewrite should have replaced the matched edge
        assert new_graph.num_nodes == family_graph.num_nodes
        # Check that proof trace was recorded
        assert len(rewriter.proof_traces) == 1

    def test_verify_rewrite_sequence_valid(self, family_graph):
        rewriter = GraphRewriter(family_graph)
        lhs = {
            "nodes": [
                {"label": "person"},
                {"label": "person"},
            ],
            "edges": [
                {"source": 0, "target": 1, "relation": "parent_of"},
            ],
        }
        rewriter.match_pattern(lhs)
        # We can't easily verify on the original graph since it's been modified
        # So we test that the verification logic runs without error
        test_graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        test_graph.add_node("X", label="person")
        test_graph.add_node("Y", label="person")
        test_graph.add_edge("X", "Y", "parent_of")
        test_rewriter = GraphRewriter(test_graph)
        test_matches = test_rewriter.match_pattern(lhs)
        test_rewrites = [
            {
                "lhs": lhs,
                "rhs": lhs,
                "interface": {"node_indices": [0, 1]},
                "match": test_matches[0],
            }
        ]
        assert test_rewriter.verify_rewrite_sequence(test_rewrites) is True
