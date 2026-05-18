"""Memory lifecycle management with topological pruning and forgetting.

Implements controlled forgetting through topological analysis, ensuring that
memory pruning preserves essential structural features of the knowledge graph.

References:
  - Edelsbrunner, H., et al. (2002). "Topological Persistence and Simplification."
    Discrete & Computational Geometry, 28(4), 511–533.
  - Parra, L., et al. (2021). "Memory for Deep Learning." NeurIPS.
"""
from __future__ import annotations

from typing import Any

from smgp.core.graph import SpectralMemoryGraph
from smgp.core.topology import TopologicalAnalyzer


class MemoryLifecycle:
    """Manages the lifecycle of knowledge graph memory.

    Periodically evaluates topological persistence of the graph and
    prunes low-importance subgraphs based on configurable forgetting
    policies.

    Attributes:
        graph: The knowledge graph to manage.
        analyzer: Topological analyzer for persistence computation.
        history: List of pruning operations for audit.

    References:
        Parra, L., et al. (2021). "Memory for Deep Learning." NeurIPS.
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        analyzer: TopologicalAnalyzer | None = None,
    ) -> None:
        """Initialize the memory lifecycle manager.

        Args:
            graph: Knowledge graph to manage.
            analyzer: Optional pre-built TopologicalAnalyzer.
        """
        self.graph = graph
        self.analyzer = analyzer or TopologicalAnalyzer(graph)
        self.history: list[dict[str, Any]] = []

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the current state of graph memory.

        Computes topological features and statistics about the graph
        to inform pruning decisions.

        Returns:
            Dict with keys:
              - 'num_nodes': Current node count.
              - 'num_edges': Current edge count.
              - 'persistence': Persistence diagram data.
              - 'betti_numbers': Betti numbers per dimension.
              - 'memory_utilization': Fraction of theoretical capacity used.
        """
        persistence = self.analyzer.compute_persistence()
        betti = persistence.get("betti_numbers", [])

        return {
            "num_nodes": self.graph.num_nodes,
            "num_edges": self.graph.num_edges,
            "persistence": persistence.get("diagrams", []),
            "betti_numbers": betti,
            "memory_utilization": self._compute_utilization(),
        }

    def prune(
        self,
        strategy: str = "low_persistence",
        threshold: float = 0.1,
    ) -> SpectralMemoryGraph:
        """Prune the knowledge graph using the specified strategy.

        Available strategies:
          - 'low_persistence': Remove nodes in low-persistence topological features.
          - 'isolated': Remove nodes with no edges.
          - 'combined': Apply both low_persistence and isolated pruning.

        Args:
            strategy: Pruning strategy name.
            threshold: Threshold parameter (strategy-dependent).

        Returns:
            New SpectralMemoryGraph with pruned nodes removed.
        """
        original_count = self.graph.num_nodes

        if strategy == "low_persistence":
            pruned = self.analyzer.prune_by_persistence(threshold)
        elif strategy == "isolated":
            pruned = self._prune_isolated()
        elif strategy == "combined":
            pruned = self.analyzer.prune_by_persistence(threshold)
            lifecycle_temp = MemoryLifecycle(pruned)
            pruned = lifecycle_temp._prune_isolated()
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

        removed = original_count - pruned.num_nodes
        self.history.append({
            "strategy": strategy,
            "threshold": threshold,
            "nodes_before": original_count,
            "nodes_after": pruned.num_nodes,
            "nodes_removed": removed,
        })

        return pruned

    def _prune_isolated(self) -> SpectralMemoryGraph:
        """Remove nodes with no edges (isolated nodes).

        Returns:
            New graph with isolated nodes removed.
        """
        node_ids = self.graph.nodes()
        non_isolated = set()
        for src, tgt, key, data in self.graph.edges():
            non_isolated.add(src)
            non_isolated.add(tgt)

        isolated = [nid for nid in node_ids if nid not in non_isolated]
        pruned = SpectralMemoryGraph(hd_dim=self.graph.hd.dim)

        for nid in node_ids:
            if nid not in isolated:
                data = self.graph.get_node(nid)
                if data:
                    pruned.add_node(nid, label=data["label"],
                                   properties=data["properties"],
                                   vector=data["vector"])

        for src, tgt, key, edata in self.graph.edges():
            pruned.add_edge(src, tgt, edata.get("relation", ""),
                           properties=edata.get("properties", {}))

        return pruned

    def _compute_utilization(self) -> float:
        """Compute memory utilization as a fraction.

        Returns:
            Float between 0 and 1 representing utilization.
        """
        # Theoretical capacity: HD vector memory
        total_capacity = 100000  # From config
        current = self.graph.num_nodes
        return min(current / max(total_capacity, 1), 1.0)
