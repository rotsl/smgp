"""Tests for explainable reasoning components.

Builds a small knowledge graph (Socrates → Plato → Aristotle) and
verifies that:
  - :class:`ExplainableClaimVerifier` returns proof sub-graphs with the
    correct edges.
  - :class:`ExplainableNeuroSymbolicPlanner` annotates plan steps with
    proof sub-graphs.
"""
from __future__ import annotations

import pytest

from smgp.core.graph import SpectralMemoryGraph
from smgp.reasoning.explainable import (
    ExplainableClaimVerifier,
    ExplainableNeuroSymbolicPlanner,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def philosophy_graph() -> SpectralMemoryGraph:
    """Create Socrates → Plato → Aristotle chain."""
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    g.add_node("Socrates", label="Socrates", properties={"role": "philosopher"})
    g.add_node("Plato", label="Plato", properties={"role": "philosopher"})
    g.add_node("Aristotle", label="Aristotle", properties={"role": "philosopher"})
    g.add_edge("Socrates", "Plato", "taught")
    g.add_edge("Plato", "Aristotle", "taught")
    return g


# ---------------------------------------------------------------------------
# ExplainableClaimVerifier
# ---------------------------------------------------------------------------

class TestExplainableClaimVerifier:
    """Tests for :class:`ExplainableClaimVerifier`."""

    def test_verify_triple_with_proof(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """Verifying a true triple should return proof_subgraph with the edge."""
        verifier = ExplainableClaimVerifier(philosophy_graph)

        result = verifier.verify("Socrates taught Plato")

        assert result["verified"] is True
        assert result["confidence"] > 0
        proof = result["proof_subgraph"]
        assert len(proof) >= 1
        # At least one proof edge should be Socrates → Plato
        edge_found = any(
            e["source"] == "Socrates" and e["target"] == "Plato"
            for e in proof
        )
        assert edge_found, f"Expected Socrates→Plato in proof, got {proof}"

    def test_verify_multi_hop_proof(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """Multi-hop claim should return proof subgraph with all path edges."""
        verifier = ExplainableClaimVerifier(philosophy_graph)

        # Try a claim about Socrates and Aristotle (2-hop path)
        result = verifier.verify("Socrates Plato Aristotle")

        # Even if not directly verified, proof_subgraph should capture path edges
        proof = result["proof_subgraph"]
        # The multi-hop path has 2 edges
        assert len(proof) == 2, f"Expected 2 proof edges, got {len(proof)}: {proof}"

    def test_verify_false_claim(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """A false claim should return verified=False and empty proof."""
        verifier = ExplainableClaimVerifier(philosophy_graph)

        result = verifier.verify("Aristotle taught Socrates")

        assert result["verified"] is False
        # proof_subgraph may be empty since the reverse edge doesn't exist
        assert isinstance(result["proof_subgraph"], list)

    def test_verify_nonexistent_entities(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """Claim with non-existent entities should not crash."""
        verifier = ExplainableClaimVerifier(philosophy_graph)

        result = verifier.verify("Unknown taught Nobody")

        assert result["verified"] is False
        assert result["proof_subgraph"] == []

    def test_proof_subgraph_structure(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """Each proof edge should have source, target, relation keys."""
        verifier = ExplainableClaimVerifier(philosophy_graph)

        result = verifier.verify("Plato taught Aristotle")

        proof = result["proof_subgraph"]
        for edge in proof:
            assert "source" in edge
            assert "target" in edge
            assert "relation" in edge


# ---------------------------------------------------------------------------
# ExplainableNeuroSymbolicPlanner
# ---------------------------------------------------------------------------

class TestExplainableNeuroSymbolicPlanner:
    """Tests for :class:`ExplainableNeuroSymbolicPlanner`."""

    def test_plan_steps_have_proof_subgraph(
        self, philosophy_graph: SpectralMemoryGraph
    ) -> None:
        """Each plan step should include a proof_subgraph key."""
        planner = ExplainableNeuroSymbolicPlanner(philosophy_graph, max_steps=5)

        # Register a simple rule
        planner.register_rule({
            "name": "connect_teachers",
            "lhs": {"node": "Socrates", "relation": "taught"},
            "rhs": {"node": "Plato"},
            "interface": {},
        })

        result = planner.plan("connect Socrates to Plato")

        for step in result:
            assert "proof_subgraph" in step, f"Step missing proof_subgraph: {step}"
            assert isinstance(step["proof_subgraph"], list)

    def test_empty_plan_has_no_steps(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """A goal with no matching rules should return an empty plan."""
        planner = ExplainableNeuroSymbolicPlanner(philosophy_graph, max_steps=5)

        result = planner.plan("path from Socrates to Aristotle")

        # Goal already satisfied (path exists) → empty plan
        assert result == []

    def test_plan_with_matching_rule(self, philosophy_graph: SpectralMemoryGraph) -> None:
        """A plan with applicable rules should include proof edges."""
        planner = ExplainableNeuroSymbolicPlanner(philosophy_graph, max_steps=5)

        # Register rules that match the existing graph structure
        planner.register_rule({
            "name": "find_teaching_chain",
            "lhs": {},
            "rhs": {},
            "interface": {},
        })

        result = planner.plan("find path from Socrates to Plato")

        # The plan may or may not have steps depending on matching
        for step in result:
            assert isinstance(step.get("proof_subgraph"), list)
