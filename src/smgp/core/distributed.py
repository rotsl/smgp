"""Distributed Memory Graph with sharding support.

Provides DistributedGraphBackend, which partitions graph data across multiple
SpectralMemoryGraph shards using consistent hashing. This enables horizontal
scaling of the knowledge graph for large datasets.

References:
  - Karger, D., et al. (1997). "Consistent Hashing and Random Trees:
    Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web."
    ACM Symposium on Theory of Computing.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from smgp.core.graph import SpectralMemoryGraph

logger = logging.getLogger(__name__)


class DistributedGraphBackend:
    """Distributed graph backend with consistent-hashing sharding.

    Partitions nodes across ``num_shards`` SpectralMemoryGraph instances using
    MD5-based consistent hashing. Cross-shard edges are stored on both
    participating shards.

    Attributes:
        num_shards: Number of graph shards.
        hd_dim: HD vector dimensionality per shard.
        shards: List of SpectralMemoryGraph instances.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_shards: int = 4,
        hd_dim: int = 10000,
        seed: int | None = None,
    ) -> None:
        """Initialize a distributed graph with multiple shards.

        Args:
            num_shards: Number of graph shards (default 4).
            hd_dim: Dimensionality of HD vectors (default 10000).
            seed: Random seed for reproducibility (passed to each shard).
        """
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        self.num_shards = num_shards
        self.hd_dim = hd_dim
        self.seed = seed
        # Give each shard a unique seed to avoid vector collisions across shards
        shard_seed = seed if seed is None else seed
        self.shards: list[SpectralMemoryGraph] = [
            SpectralMemoryGraph(
                hd_dim=hd_dim,
                seed=(shard_seed + i) if shard_seed is not None else None,
            )
            for i in range(num_shards)
        ]

    def _get_shard(self, node_id: str) -> int:
        """Determine the shard index for a given node ID using consistent hashing.

        Uses MD5 hash of the node ID, mapped to the shard ring via modulo.

        Args:
            node_id: The node identifier string.

        Returns:
            Integer shard index in [0, num_shards).
        """
        digest = hashlib.md5(node_id.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.num_shards

    def add_node(
        self,
        node_id: str,
        label: str = "",
        properties: dict[str, Any] | None = None,
        vector: np.ndarray | None = None,
    ) -> str:
        """Add a node to the appropriate shard.

        Args:
            node_id: Unique node identifier.
            label: Semantic type label.
            properties: Optional metadata dictionary.
            vector: Optional pre-computed HD vector.

        Returns:
            The node_id of the added node.
        """
        shard_idx = self._get_shard(node_id)
        self.shards[shard_idx].add_node(
            node_id, label=label, properties=properties, vector=vector
        )
        logger.debug("Added node '%s' to shard %d", node_id, shard_idx)
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add an edge, routing to the appropriate shard(s).

        If source and target are on different shards, the edge is added to
        both shards to ensure local traversal works.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Relation type string.
            properties: Optional edge metadata.

        Returns:
            The edge key (UUID string).
        """
        source_shard = self._get_shard(source_id)
        target_shard = self._get_shard(target_id)

        # Ensure primary nodes exist in their respective shards
        if source_id not in self.shards[source_shard].graph:
            raise KeyError(
                f"Source node '{source_id}' not found in shard {source_shard}."
            )
        if target_id not in self.shards[target_shard].graph:
            raise KeyError(
                f"Target node '{target_id}' not found in shard {target_shard}."
            )

        if source_shard == target_shard:
            # Same shard: simple edge addition
            edge_key = self.shards[source_shard].add_edge(
                source_id, target_id, relation, properties=properties
            )
        else:
            # Cross-shard: replicate missing nodes so the edge can exist on both shards
            # Ensure target node exists on source shard
            if target_id not in self.shards[source_shard].graph:
                tgt_data = self.shards[target_shard].get_node(target_id)
                self.shards[source_shard].add_node(
                    target_id,
                    label=tgt_data["label"] if tgt_data else "",
                    properties=tgt_data["properties"] if tgt_data else {},
                    vector=tgt_data["vector"] if tgt_data else None,
                )
            # Ensure source node exists on target shard
            if source_id not in self.shards[target_shard].graph:
                src_data = self.shards[source_shard].get_node(source_id)
                self.shards[target_shard].add_node(
                    source_id,
                    label=src_data["label"] if src_data else "",
                    properties=src_data["properties"] if src_data else {},
                    vector=src_data["vector"] if src_data else None,
                )
            # Add edge to both shards
            edge_key = self.shards[source_shard].add_edge(
                source_id, target_id, relation, properties=properties
            )
            self.shards[target_shard].add_edge(
                source_id, target_id, relation, properties=properties
            )

        logger.debug(
            "Added edge '%s' -> '%s' (relation='%s')",
            source_id, target_id, relation,
        )
        return edge_key

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve node data from the correct shard.

        Args:
            node_id: Node identifier.

        Returns:
            Dict with keys 'label', 'properties', 'vector', or None.
        """
        shard_idx = self._get_shard(node_id)
        return self.shards[shard_idx].get_node(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from its shard.

        Args:
            node_id: Node identifier.

        Raises:
            KeyError: If node not found in the expected shard.
        """
        shard_idx = self._get_shard(node_id)
        self.shards[shard_idx].remove_node(node_id)
        logger.debug("Removed node '%s' from shard %d", node_id, shard_idx)

    def query_similar(
        self,
        vector: np.ndarray,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Query all shards and merge results for top-k similar nodes.

        Args:
            vector: Query HD vector.
            k: Number of results.

        Returns:
            List of (node_id, similarity_score) tuples, sorted descending.
        """
        all_results: list[tuple[str, float]] = []

        for i, shard in enumerate(self.shards):
            shard_results = shard.query_similar(vector, k=k)
            all_results.extend(shard_results)

        # Sort by similarity and return top-k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

    @property
    def num_nodes(self) -> int:
        """Total number of nodes across all shards.

        Note: This may overcount if cross-shard edges caused node duplication
        on target shards.
        """
        return sum(shard.num_nodes for shard in self.shards)

    @property
    def num_edges(self) -> int:
        """Total number of edges across all shards."""
        return sum(shard.num_edges for shard in self.shards)

    def nodes(self) -> list[str]:
        """Return aggregated list of unique node IDs from all shards."""
        all_nodes: list[str] = []
        seen: set = set()
        for shard in self.shards:
            for node_id in shard.nodes():
                if node_id not in seen:
                    all_nodes.append(node_id)
                    seen.add(node_id)
        return all_nodes

    def edges(self) -> list[tuple[str, str, str, dict[str, Any]]]:
        """Return aggregated list of unique edges from all shards."""
        all_edges: list[tuple[str, str, str, dict[str, Any]]] = []
        seen_keys: set = set()
        for shard in self.shards:
            for edge in shard.edges():
                key = (edge[0], edge[1], edge[2])
                if key not in seen_keys:
                    all_edges.append(edge)
                    seen_keys.add(key)
        return all_edges

    def shard_distribution(self) -> dict[int, int]:
        """Return the number of primary nodes per shard.

        Returns:
            Dict mapping shard index to node count.
        """
        return {i: shard.num_nodes for i, shard in enumerate(self.shards)}

    def __repr__(self) -> str:
        return (
            f"DistributedGraphBackend("
            f"num_shards={self.num_shards}, "
            f"hd_dim={self.hd_dim}, "
            f"total_nodes={self.num_nodes}, "
            f"total_edges={self.num_edges})"
        )
