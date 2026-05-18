"""Basic memory example: building a knowledge graph and querying it."""
import numpy as np
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.hyperdim import HyperdimensionalMemory
from smgp.core.spectral import SpectralMethods


def main():
    # Create a knowledge graph
    graph = SpectralMemoryGraph(hd_dim=10000, seed=42)

    # Add knowledge
    graph.add_node("SMGP", label="system", properties={"type": "AI"})
    graph.add_node("graph_theory", label="field", properties={"domain": "mathematics"})
    graph.add_node("spectral_analysis", label="technique")
    graph.add_node("persistent_memory", label="feature")

    graph.add_edge("SMGP", "graph_theory", "based_on")
    graph.add_edge("SMGP", "spectral_analysis", "uses")
    graph.add_edge("SMGP", "persistent_memory", "provides")

    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Query by HD similarity
    query_vec = graph.get_node("SMGP")["vector"]
    similar = graph.query_similar(query_vec, k=3)
    print("\nSimilar nodes:")
    for nid, sim in similar:
        print(f"  {nid}: {sim:.4f}")

    # Spectral analysis
    if graph.num_nodes >= 2:
        spectral = SpectralMethods(graph, num_eigenvalues=3)
        L = spectral.compute_laplacian()
        eigenvalues, eigenvectors = spectral.compute_eigen()
        print(f"\nEigenvalues: {eigenvalues}")

    # HD vector operations
    hd = graph.hd
    v1 = hd.generate(1)[0]
    v2 = hd.generate(1)[0]
    bound = hd.bind(v1, v2)
    unbound = hd.unbind(bound, v2)
    recovered = hd.similarity(v1, unbound)
    print(f"\nBind/Unbind recovery: {recovered:.6f} (should be 1.0)")


if __name__ == "__main__":
    main()
