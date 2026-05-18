# API Reference

Complete reference for all SMGP classes and public methods.

**Repository:** [https://github.com/rotsl/smgp](https://github.com/rotsl/smgp)

---

## Core Module: `smgp.core.graph`

### `SpectralMemoryGraph`

The central data structure — a typed property graph with hyperdimensional node addressing.

```python
from smgp.core.graph import SpectralMemoryGraph
```

#### Constructor

```python
graph = SpectralMemoryGraph(
    hd_dim: int = 10000,       # Dimensionality of HD vectors
    seed: int | None = None,   # Random seed for reproducibility
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_node` | `(node_id: str, label: str = "", properties: dict = None)` | Add a node with an auto-generated HD vector |
| `get_node` | `(node_id: str) -> dict` | Get node data including HD vector |
| `remove_node` | `(node_id: str)` | Remove a node and all its edges |
| `add_edge` | `(source: str, target: str, relation: str, properties: dict = None)` | Add a typed edge between two nodes |
| `get_edge` | `(source: str, target: str) -> dict` | Get edge data |
| `remove_edge` | `(source: str, target: str)` | Remove an edge |
| `get_neighbors` | `(node_id: str) -> list[str]` | Get adjacent node IDs |
| `get_edges` | `(node_id: str, relation: str = None) -> list[tuple]` | Get edges from/to a node, optionally filtered by relation |
| `query_similar` | `(vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]` | Find k most similar nodes by HD cosine similarity |
| `shortest_path` | `(source: str, target: str) -> list[str]` | Find shortest path between two nodes |
| `find_paths` | `(source: str, target: str, max_length: int = 5) -> list[list[str]]` | Find all paths up to max_length |
| `subgraph` | `(node_ids: list[str]) -> SpectralMemoryGraph` | Extract a subgraph |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `nodes` | `list[str]` | All node IDs |
| `num_nodes` | `int` | Number of nodes |
| `num_edges` | `int` | Number of edges |
| `hd` | `HyperdimensionalMemory` | The HD memory subsystem |

#### Class Methods

| Method | Description |
|--------|-------------|
| `SpectralMemoryGraph.from_config(config)` | Create from `SMGPConfig` instance |

---

## Core Module: `smgp.core.hyperdim`

### `HyperdimensionalMemory`

Manages hyperdimensional (bipolar) vector generation, binding, and similarity operations.

```python
from smgp.core.hyperdim import HyperdimensionalMemory
```

#### Constructor

```python
hd = HyperdimensionalMemory(
    dim: int = 10000,          # Vector dimensionality
    seed: int | None = None,   # Random seed
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `generate` | `(n: int = 1) -> np.ndarray` | Generate n random bipolar HD vectors |
| `bind` | `(v1: np.ndarray, v2: np.ndarray) -> np.ndarray` | Bind two vectors (element-wise multiplication) |
| `unbind` | `(bound: np.ndarray, v: np.ndarray) -> np.ndarray` | Unbind a vector from a bound pair |
| `bundle` | `(vectors: list[np.ndarray]) -> np.ndarray` | Bundle (superpose) multiple vectors |
| `similarity` | `(v1: np.ndarray, v2: np.ndarray) -> float` | Cosine similarity between two HD vectors |
| `similarity_search` | `(query: np.ndarray, candidates: np.ndarray, k: int = 5)` | Find k most similar vectors |
| `encode_string` | `(s: str) -> np.ndarray` | Encode a string as an HD vector |
| `permute` | `(v: np.ndarray, n: int = 1) -> np.ndarray` | Apply cyclic permutation n times |

---

## Core Module: `smgp.core.spectral`

### `SpectralMethods`

Spectral analysis of the knowledge graph: Laplacian, eigendecomposition, Fourier transform.

```python
from smgp.core.spectral import SpectralMethods
```

#### Constructor

```python
spectral = SpectralMethods(
    graph: SpectralMemoryGraph,
    num_eigenvalues: int = 64,  # Number of eigenvalues to compute
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_laplacian` | `(normalized: bool = True) -> np.ndarray` | Compute graph Laplacian matrix |
| `compute_eigen` | `() -> tuple[np.ndarray, np.ndarray]` | Compute eigenvalues and eigenvectors |
| `graph_fourier_transform` | `(signal: list[float]) -> np.ndarray` | Apply graph Fourier transform |
| `inverse_fourier_transform` | `(coefficients: np.ndarray) -> np.ndarray` | Apply inverse graph Fourier transform |
| `spectral_filter` | `(signal, filter_func)` | Apply a spectral filter to a graph signal |
| `spectral_clustering` | `(n_clusters: int) -> np.ndarray` | Cluster nodes via spectral clustering |
| `compute_wavelet` | `(signal, scale: float)` | Compute graph wavelet transform at given scale |

---

## Core Module: `smgp.core.topology`

### `TopologicalAnalyzer`

Topological data analysis of the knowledge graph using persistent homology.

```python
from smgp.core.topology import TopologicalAnalyzer
```

#### Constructor

```python
topo = TopologicalAnalyzer(
    graph: SpectralMemoryGraph,
    max_dimension: int = 2,           # Max simplicial dimension
    persistence_threshold: float = 0.1,  # Threshold for significant features
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_persistence` | `() -> list[dict]` | Compute persistent homology |
| `betti_numbers` | `() -> dict[int, int]` | Compute Betti numbers β_0, β_1, β_2 |
| `identify_persistent_nodes` | `() -> set[str]` | Identify nodes in persistent topological features |
| `identify_eviction_candidates` | `(age_threshold: float = 1.0) -> list[str]` | Find nodes safe to evict |

---

## Core Module: `smgp.core.category`

### `GraphRewriter`

Category-theoretic graph rewriting using Double Pushout (DPO) transformation.

```python
from smgp.core.category import GraphRewriter
```

#### Constructor

```python
rewriter = GraphRewriter(
    graph: SpectralMemoryGraph,
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `match_pattern` | `(pattern: dict) -> list[dict]` | Find all matches of a graph pattern |
| `apply_rule` | `(pattern: dict, replacement: dict) -> bool` | Apply a DPO rewrite rule |
| `merge_nodes` | `(node_ids: list[str]) -> str` | Merge multiple nodes into one |

---

## Memory Module: `smgp.memory`

### `MemoryStore`

Thread-safe persistent key-value store with LRU eviction.

```python
from smgp.memory.store import MemoryStore

store = MemoryStore(max_size=10000)
store.put("key", {"data": "value"})
result = store.get("key")
```

### `AssociativeMemory`

Content-addressable retrieval by hyperdimensional similarity.

```python
from smgp.memory.associative import AssociativeMemory

mem = AssociativeMemory(hd_dim=10000, seed=42)
mem.store("Socrates", {"label": "person"})
results = mem.retrieve(query_vector, k=5)
```

### `MemoryLifecycle`

Manages memory persistence, consolidation, and forgetting using topological analysis.

```python
from smgp.memory.lifecycle import MemoryLifecycle

lifecycle = MemoryLifecycle(graph, persistence_threshold=0.1, age_threshold=30.0)
lifecycle.tick()  # Run one lifecycle step
evicted = lifecycle.get_evicted()  # Get recently evicted items
```

---

## Attention Module: `smgp.attention`

### `SpectralAttention`

O(N log N) multi-head, multi-scale spectral attention mechanism.

```python
from smgp.attention.spectral_attn import SpectralAttention

attn = SpectralAttention(
    graph: SpectralMemoryGraph,
    hidden_dim: int = 64,
    num_heads: int = 8,
    num_scales: int = 4,
    temperature: float = 1.0,
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `build_graph_from_tokens` | `(tokens: np.ndarray) -> None` | Build a context graph from token embeddings |
| `forward` | `(tokens: np.ndarray) -> np.ndarray` | Apply spectral attention to input tokens |
| `hierarchical_coarsening` | `() -> list[SpectralMemoryGraph]` | Create multi-scale graph pyramid |

---

## Reasoning Module: `smgp.reasoning`

### `ClaimVerifier`

Verify factual claims against the knowledge graph using path-based reasoning.

```python
from smgp.reasoning.verifier import ClaimVerifier

verifier = ClaimVerifier(graph)
result = verifier.verify("Socrates taught Plato")
# Returns: {"verified": True, "reasoning": "...", "confidence": 0.95}
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `verify` | `(claim: str) -> dict` | Verify a single claim |
| `verify_batch` | `(claims: list[str]) -> list[dict]` | Verify multiple claims |

### `NeuroSymbolicPlanner`

Generate reasoning plans grounded in graph facts.

```python
from smgp.reasoning.planner import NeuroSymbolicPlanner

planner = NeuroSymbolicPlanner(graph)
plan = planner.plan(query="How to treat fever?", max_depth=3)
# Returns: {"steps": [...], "conclusion": "...", "confidence": 0.85}
```

---

## Integration Module: `smgp.integration`

### HuggingFace: `SMGPForCausalLM`

```python
from smgp.integration.huggingface import SMGPForCausalLM

model = SMGPForCausalLM(graph_hd_dim=10000)
model.add_knowledge("Socrates", "person")
model.add_knowledge("Socrates", "Plato", "taught")
output = model.generate(prompt="Who did Socrates teach?", max_length=50)
```

### LangChain: `SMGPMemory` and `SMGPVerifierTool`

```python
from smgp.integration.langchain import SMGPMemory, SMGPVerifierTool

memory = SMGPMemory(hd_dim=10000)
memory.save_context({"input": "..."}, {"output": "..."})
context = memory.load_memory_variables({"query": "..."})

tool = SMGPVerifierTool(graph=memory.graph)
result = tool.run("Socrates taught Plato")
```

### REST API

```python
from smgp.integration.api import create_app

app = create_app()
# Run with: uvicorn smgp.integration.api:app --host 0.0.0.0 --port 8000
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/graph/nodes` | Add node |
| `POST` | `/graph/edges` | Add edge |
| `GET` | `/graph/nodes/{id}` | Get node |
| `POST` | `/query` | Query similar nodes |
| `POST` | `/verify` | Verify a claim |
| `POST` | `/reason` | Generate reasoning plan |
| `GET` | `/stats` | Graph statistics |

---

## Utilities: `smgp.utils`

### `save_graph` / `load_graph`

```python
from smgp.utils.io import save_graph, load_graph

save_graph(graph, "path/to/graph.json")
graph = load_graph("path/to/graph.json")
```

### Benchmarking CLI

```bash
smgp bench --num-nodes 1000 --num-queries 100
```
