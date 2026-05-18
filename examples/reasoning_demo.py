"""Reasoning demo: neuro-symbolic planning and claim verification."""
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.category import GraphRewriter
from smgp.reasoning.verifier import ClaimVerifier
from smgp.reasoning.planner import NeuroSymbolicPlanner


def main():
    # Build a knowledge graph with facts
    graph = SpectralMemoryGraph(hd_dim=1000, seed=42)

    # Add entities
    graph.add_node("Socrates", label="person", properties={"name": "Socrates"})
    graph.add_node("Plato", label="person", properties={"name": "Plato"})
    graph.add_node("Aristotle", label="person", properties={"name": "Aristotle"})
    graph.add_node("philosophy", label="field", properties={"name": "philosophy"})
    graph.add_node("Academy", label="institution", properties={"name": "Academy"})

    # Add relations
    graph.add_edge("Socrates", "philosophy", "studied")
    graph.add_edge("Socrates", "Plato", "taught")
    graph.add_edge("Plato", "philosophy", "studied")
    graph.add_edge("Plato", "Aristotle", "taught")
    graph.add_edge("Plato", "Academy", "founded")

    print("Knowledge Graph:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")

    # Verify claims
    verifier = ClaimVerifier(graph)

    claims = [
        "Socrates taught Plato",
        "Plato founded Academy",
        "Socrates founded Academy",
    ]

    print("\nClaim Verification:")
    for claim in claims:
        result = verifier.verify(claim)
        status = "VERIFIED" if result["verified"] else "UNVERIFIED"
        print(f"  [{status}] {claim} (confidence={result['confidence']:.2f})")

    # Graph rewriting demo
    print("\nGraph Rewriting (DPO):")
    rewriter = GraphRewriter(graph)

    # Pattern: person X taught person Y
    pattern = {
        "nodes": [
            {"label": "person"},
            {"label": "person"},
        ],
        "edges": [
            {"source": 0, "target": 1, "relation": "taught"},
        ],
    }

    matches = rewriter.match_pattern(pattern)
    print(f"  Found {len(matches)} 'taught' relations:")
    for match in matches:
        print(f"    {match['0']} -> {match['1']}")


if __name__ == "__main__":
    main()
