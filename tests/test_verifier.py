"""Tests for the ClaimVerifier module."""
import pytest

from smgp.core.graph import SpectralMemoryGraph
from smgp.reasoning.verifier import ClaimVerifier


@pytest.fixture
def verifier_graph():
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    g.add_node("Paris", label="city", properties={"name": "Paris"})
    g.add_node("France", label="country", properties={"name": "France"})
    g.add_node("London", label="city", properties={"name": "London"})
    g.add_node("UK", label="country", properties={"name": "UK"})
    g.add_edge("Paris", "France", "capital_of")
    g.add_edge("London", "UK", "capital_of")
    return g


class TestClaimVerifier:
    def test_verify_direct_edge(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        result = verifier.verify("Paris capital_of France")
        assert result["verified"] is True
        assert result["confidence"] > 0

    def test_verify_no_path(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        result = verifier.verify("Tokyo is the capital of Japan")
        assert result["verified"] is False
        assert result["confidence"] == 0.0

    def test_verify_is_pattern(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        result = verifier.verify("Paris is the capital of France")
        # "is the" is a parsed pattern but won't match "capital_of" directly
        # The verify function should still run without error
        assert "verified" in result
        assert "confidence" in result

    def test_find_path_exists(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        path = verifier.find_path("Paris", "France")
        assert path is not None
        assert len(path) >= 2

    def test_find_path_no_exists(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        path = verifier.find_path("Paris", "NONEXISTENT")
        assert path is None

    def test_parse_claim(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        parsed = verifier._parse_claim("Paris is France")
        assert parsed is not None
        assert parsed["subject"] == "Paris"
        assert parsed["predicate"] == "is"
        assert parsed["object"] == "France"

    def test_parse_claim_complex(self, verifier_graph):
        verifier = ClaimVerifier(verifier_graph)
        parsed = verifier._parse_claim("Paris is the capital of France")
        assert parsed is not None
        assert parsed["subject"] == "Paris"
