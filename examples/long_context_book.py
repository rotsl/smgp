"""Long context processing demo: processing a synthetic book chapter."""
import numpy as np
from smgp.core.graph import SpectralMemoryGraph
from smgp.attention.spectral_attn import SpectralAttention


def main():
    # Simulate a chapter of 256 tokens with 64-dim embeddings
    seq_len = 256
    hidden_dim = 64

    np.random.seed(42)
    tokens = np.random.randn(seq_len, hidden_dim) * 0.1

    # Add some structure: make tokens 0-50 similar to tokens 200-256
    theme_vector = np.random.randn(hidden_dim) * 0.5
    tokens[:50] += theme_vector * 0.3
    tokens[200:] += theme_vector * 0.3

    # Build graph and run spectral attention
    graph = SpectralMemoryGraph(hd_dim=100, seed=42)
    attn = SpectralAttention(
        graph,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_scales=3,
    )

    print("Building context graph...")
    attn.build_graph_from_tokens(tokens)
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")

    print("Running spectral attention...")
    output = attn.forward(tokens)
    print(f"Output shape: {output.shape}")

    # Check long-range dependency
    similarity = np.dot(output[0], output[-1]) / (
        np.linalg.norm(output[0]) * np.linalg.norm(output[-1])
    )
    print(f"First-last token similarity after attention: {similarity:.4f}")

    # Hierarchical coarsening
    print("\nHierarchical coarsening:")
    levels = attn.hierarchical_coarsening()
    for i, level in enumerate(levels):
        print(f"  Level {i}: {level.num_nodes} nodes")


if __name__ == "__main__":
    main()
