# Quick Start Guide

Get up and running with SMGP in under 5 minutes.

## Installation

```bash
# Basic installation
pip install smgp

# With all optional dependencies
pip install "smgp[all]"
```

Or install from source:

```bash
git clone https://github.com/rotsl/smgp.git
cd smgp
pip install -e ".[dev]"
```

## Your First Knowledge Graph

Create a knowledge graph, add nodes and edges, and query by similarity:

```python
from smgp.core.graph import SpectralMemoryGraph

# Create a graph with 10,000-dimensional hyperdimensional addressing
graph = SpectralMemoryGraph(hd_dim=10000, seed=42)

# Add nodes with labels and optional properties
graph.add_node("Socrates", label="person", properties={"era": "ancient Greece"})
graph.add_node("Plato", label="person", properties={"era": "ancient Greece"})
graph.add_node("Aristotle", label="person", properties={"era": "ancient Greece"})
graph.add_node("philosophy", label="field", properties={"domain": "humanities"})
graph.add_node("Academy", label="institution")

# Add typed edges to encode relationships
graph.add_edge("Socrates", "Plato", "taught")
graph.add_edge("Socrates", "philosophy", "studied")
graph.add_edge("Plato", "Aristotle", "taught")
graph.add_edge("Plato", "philosophy", "studied")
graph.add_edge("Plato", "Academy", "founded")

print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
```

## Querying by Similarity

Find nodes similar to a query using hyperdimensional vector similarity:

```python
# Get the HD vector for a known node
query_vec = graph.get_node("Socrates")["vector"]

# Find the top-k most similar nodes
similar = graph.query_similar(query_vec, k=5)
for node_id, score in similar:
    print(f"  {node_id}: similarity={score:.4f}")
```

## Spectral Analysis

Compute graph Laplacian, eigendecomposition, and Fourier transforms:

```python
from smgp.core.spectral import SpectralMethods

# Create spectral analyzer
spectral = SpectralMethods(graph, num_eigenvalues=5)

# Compute normalized Laplacian
L = spectral.compute_laplacian(normalized=True)
print(f"Laplacian shape: {L.shape}")

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = spectral.compute_eigen()
print(f"Eigenvalues: {eigenvalues}")

# Graph Fourier transform a signal
signal = [1.0 if i % 2 == 0 else 0.0 for i in range(graph.num_nodes)]
fourier = spectral.graph_fourier_transform(signal)
print(f"Fourier coefficients: {fourier}")
```

## Claim Verification

Verify factual claims against the knowledge graph:

```python
from smgp.reasoning.verifier import ClaimVerifier

verifier = ClaimVerifier(graph)

# Verify individual claims
claims = [
    "Socrates taught Plato",
    "Plato founded Academy",
    "Socrates founded Academy",     # False — Plato did
    "Aristotle taught Plato",       # False — reverse direction
]

for claim in claims:
    result = verifier.verify(claim)
    status = "VERIFIED" if result["verified"] else "UNVERIFIED"
    print(f"[{status}] {claim}")
    print(f"  Reasoning: {result['reasoning']}")
```

## Neuro-Symbolic Planning

Generate reasoning plans grounded in graph facts:

```python
from smgp.reasoning.planner import NeuroSymbolicPlanner

planner = NeuroSymbolicPlanner(graph)

plan = planner.plan(
    query="Who was influenced by Socrates?",
    max_depth=3,
    max_branching=2,
)

for step in plan["steps"]:
    print(f"Step {step['step']}: {step['action']}")
    print(f"  Evidence: {step.get('evidence', 'N/A')}")

print(f"Conclusion: {plan.get('conclusion', 'N/A')}")
```

## Hyperdimensional Memory

Work directly with HD vectors for custom applications:

```python
from smgp.core.hyperdim import HyperdimensionalMemory

hd = HyperdimensionalMemory(dim=10000, seed=42)

# Generate random HD vectors
vectors = hd.generate(5)

# Bind (pair) and unbind (retrieve)
v1, v2 = hd.generate(2)
bound = hd.bind(v1, v2)
recovered = hd.unbind(bound, v2)
similarity = hd.similarity(v1, recovered)
print(f"Recovery similarity: {similarity:.6f}  (should be ~1.0)")

# Bundle (superposition)
bundled = hd.bundle(vectors)

# Similarity search
candidates = hd.generate(1000)
query = candidates[42].copy()
top5 = hd.similarity_search(query, candidates, k=5)
print(f"Top-5 matches for index 42: {[idx for idx, _ in top5]}")
```

## Long-Context Processing

Use spectral attention for efficient long-sequence processing:

```python
import numpy as np
from smgp.core.graph import SpectralMemoryGraph
from smgp.attention.spectral_attn import SpectralAttention

# Simulate 512 tokens with 64-dim embeddings
tokens = np.random.randn(512, 64).astype(np.float32) * 0.1

# Build spectral attention
graph = SpectralMemoryGraph(hd_dim=100, seed=42)
attn = SpectralAttention(graph, hidden_dim=64, num_heads=4, num_scales=3)

# Build context graph from tokens
attn.build_graph_from_tokens(tokens)

# Run spectral attention — O(N log N) instead of O(N²)
output = attn.forward(tokens)
print(f"Output shape: {output.shape}")

# Hierarchical coarsening for multi-scale processing
levels = attn.hierarchical_coarsening()
for i, level in enumerate(levels):
    print(f"Level {i}: {level.num_nodes} nodes")
```

## Saving and Loading Graphs

Persist your knowledge graph to disk:

```python
from smgp.utils.io import save_graph, load_graph

# Save to file
save_graph(graph, "my_knowledge.json")

# Load from file
loaded = load_graph("my_knowledge.json")
print(f"Loaded graph: {loaded.num_nodes} nodes")
```

## Next Steps

- Read the [API Reference](api.md) for full class and method documentation
- Check the [examples/](../examples/) directory for complete demo scripts
- Read [RESEARCH.md](../RESEARCH.md) for mathematical foundations
- Explore integrations: [HuggingFace](../README.md#huggingface-transformers), [LangChain](../README.md#langchain), [REST API](../README.md#rest-api)
