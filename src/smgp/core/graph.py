"""Persistent, typed, multi-relational knowledge graph with HD vector addressing.

The SpectralMemoryGraph is the central data structure of SMGP. It is a directed,
multi-relational, typed property graph where every node carries a hyperdimensional
vector (HD vector) that serves as its content address. Edges are typed relations
with optional properties.

Graph operations support:
  - Adding and removing nodes/edges
  - Querying by HD vector similarity (O(N) scan, O(1) with GPU/HD hardware)
  - Extracting subgraphs
  - Computing adjacency matrices for spectral analysis

References:
  - Angles, R., & Gutierrez, C. (2008). "Survey of Graph Database Models."
    ACM Computing Surveys, 40(1), 1–39.
  - Miller, J.A. (2010). "Effects of Dimensionality on the Holographic Reduced
    Representation." Neurocomputing, 73(4-6), 593–600.
"""
from __future__ import annotations

import uuid
from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse

from smgp.core.hyperdim import HyperdimensionalMemory

try:
    from smgp.hw_bridge import HardwareBridge as _HardwareBridge
except ImportError:  # hardware directory not on path
    _HardwareBridge = None  # type: ignore[assignment,misc]


class SpectralMemoryGraph:
    """Directed, multi-relational property graph with hyperdimensional addressing.

    Each node is identified by a unique string ID and carries:
      - A text label (e.g., "person", "concept", "fact")
      - An optional dict of properties
      - A D-dimensional bipolar HD vector for content-addressable retrieval

    Edges are directed, typed relations (e.g., "is_a", "has_property") between
    nodes, stored in a NetworkX MultiDiGraph to support parallel edges.

    Attributes:
        hd: HyperdimensionalMemory engine for vector operations.
        graph: Internal NetworkX MultiDiGraph.

    References:
        Angles, R. & Gutierrez, C. (2008). "Survey of Graph Database Models."
    """

    def __init__(self, hd_dim: int = 10000, seed: int | None = None,
                 executor: Any = None) -> None:
        """Initialize an empty knowledge graph.

        Args:
            hd_dim: Dimensionality of HD vectors per node.
            seed: Random seed for reproducible HD vector generation.
            executor: Optional hardware executor (e.g. ``HWExecutor``).
                When provided, supported operations are offloaded to the
                accelerator.  Defaults to ``None`` (pure-Python).
        """
        self.hd = HyperdimensionalMemory(dim=hd_dim, seed=seed, executor=executor)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.executor = executor
        self.hw_bridge = (
            _HardwareBridge(executor)
            if executor is not None and _HardwareBridge is not None
            else None
        )

    def add_node(
        self,
        node_id: str,
        label: str = "",
        properties: dict[str, Any] | None = None,
        vector: np.ndarray | None = None,
    ) -> str:
        """Add a node to the graph.

        Args:
            node_id: Unique string identifier for the node.
            label: Semantic type label (e.g., "entity", "concept").
            properties: Optional metadata dictionary.
            vector: Optional pre-computed HD vector; if None, one is generated.

        Returns:
            The node_id of the added node.
        """
        if vector is None:
            vector = self.hd.generate(1)[0]
        self.graph.add_node(
            node_id,
            label=label,
            properties=properties or {},
            vector=vector,
        )
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add a directed, typed edge between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relation: Relation type string (e.g., "is_a", "causes").
            properties: Optional edge metadata.

        Returns:
            The edge key (UUID string).

        Raises:
            KeyError: If source_id or target_id does not exist in the graph.
        """
        if source_id not in self.graph:
            raise KeyError(f"Source node '{source_id}' not found in graph.")
        if target_id not in self.graph:
            raise KeyError(f"Target node '{target_id}' not found in graph.")
        edge_key = str(uuid.uuid4())
        self.graph.add_edge(
            source_id,
            target_id,
            key=edge_key,
            relation=relation,
            properties=properties or {},
        )
        return edge_key

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges.

        Args:
            node_id: ID of the node to remove.

        Raises:
            KeyError: If node_id does not exist.
        """
        if node_id not in self.graph:
            raise KeyError(f"Node '{node_id}' not found in graph.")
        self.graph.remove_node(node_id)

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge by its key.

        Args:
            edge_id: Edge key (UUID string) to remove.

        Raises:
            KeyError: If no edge with the given key exists.
        """
        for u, v, k, data in list(self.graph.edges(keys=True, data=True)):
            if k == edge_id:
                self.graph.remove_edge(u, v, k)
                return
        raise KeyError(f"Edge '{edge_id}' not found in graph.")

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve node data.

        Args:
            node_id: Node identifier.

        Returns:
            Dict with keys 'label', 'properties', 'vector', or None if not found.
        """
        if node_id not in self.graph:
            return None
        data = self.graph.nodes[node_id]
        return {
            "label": data.get("label", ""),
            "properties": data.get("properties", {}),
            "vector": data.get("vector"),
        }

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        """Retrieve edge data by key.

        Args:
            edge_id: Edge key (UUID).

        Returns:
            Dict with keys 'source', 'target', 'relation', 'properties', or None.
        """
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if k == edge_id:
                return {
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", ""),
                    "properties": data.get("properties", {}),
                }
        return None

    def query_similar(
        self,
        vector: np.ndarray,
        k: int = 5,
        node_type: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find k most similar nodes by HD vector cosine similarity.

        Args:
            vector: Query HD vector.
            k: Number of results.
            node_type: Optional label filter.

        Returns:
            List of (node_id, similarity_score) tuples, sorted descending.
        """
        candidates: list[tuple[str, float]] = []
        for nid, data in self.graph.nodes(data=True):
            if node_type and data.get("label") != node_type:
                continue
            sim = self.hd.similarity(vector, data["vector"])
            candidates.append((nid, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def subgraph(self, node_ids: list[str]) -> SpectralMemoryGraph:
        """Extract a subgraph containing only the specified nodes and their edges.

        Args:
            node_ids: List of node IDs to include.

        Returns:
            New SpectralMemoryGraph containing only the selected nodes and edges.
        """
        sub_nodes = {nid for nid in node_ids if nid in self.graph}
        induced = self.graph.subgraph(sub_nodes).copy()
        new_g = SpectralMemoryGraph(hd_dim=self.hd.dim)
        for nid, data in induced.nodes(data=True):
            new_g.graph.add_node(nid, **data)
        for u, v, k, data in induced.edges(keys=True, data=True):
            new_g.graph.add_edge(u, v, key=k, **data)
        return new_g

    def adjacency_matrix(self) -> sparse.csr_matrix:
        """Compute the weighted adjacency matrix.

        For the spectral Laplacian computation, we use an unweighted adjacency
        matrix where A[i,j] = 1 if there exists at least one edge from node i
        to node j.

        Returns:
            scipy.sparse.csr_matrix of shape (N, N) where N = num_nodes.
        """
        node_list = list(self.graph.nodes())
        n = len(node_list)
        if n == 0:
            return sparse.csr_matrix((0, 0))
        adj = nx.to_scipy_sparse_array(
            self.graph, nodelist=node_list, format="csr", weight=None
        )
        # Make binary (at least one edge -> 1)
        adj.data[:] = 1
        return adj

    def node_vectors(self) -> dict[str, np.ndarray]:
        """Get a dict mapping node IDs to their HD vectors.

        Returns:
            Dictionary of {node_id: np.ndarray}.
        """
        return {
            nid: data["vector"]
            for nid, data in self.graph.nodes(data=True)
            if "vector" in data
        }

    def node_to_index(self) -> dict[str, int]:
        """Map node IDs to contiguous integer indices.

        Returns:
            Dictionary mapping node_id -> integer index.
        """
        return {nid: i for i, nid in enumerate(self.graph.nodes())}

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.graph.number_of_edges()

    def nodes(self) -> list[str]:
        """Return a list of all node IDs."""
        return list(self.graph.nodes())

    def edges(self) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Return a list of all edges as (source, target, key, data) tuples."""
        return list(self.graph.edges(keys=True, data=True))

    def neighbors(self, node_id: str) -> list[str]:
        """Return list of successor node IDs."""
        if node_id not in self.graph:
            return []
        return list(self.graph.successors(node_id))

    def predecessors(self, node_id: str) -> list[str]:
        """Return list of predecessor node IDs."""
        if node_id not in self.graph:
            return []
        return list(self.graph.predecessors(node_id))
