"""Enhanced memory lifecycle manager with pluggable pruning policies.

Extends :class:`~smgp.memory.lifecycle.MemoryLifecycle` to accept an
arbitrary :class:`~smgp.memory.prune_policies.PrunePolicy`.  When a
policy is provided the :meth:`prune` method delegates node selection to
the policy instead of using the built-in strategies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from smgp.core.graph import SpectralMemoryGraph
from smgp.memory.lifecycle import MemoryLifecycle
from smgp.memory.prune_policies import PrunePolicy

if TYPE_CHECKING:
    pass


class EnhancedMemoryLifecycle(MemoryLifecycle):
    """Memory lifecycle manager with pluggable pruning policies.

    Parameters
    ----------
    graph : SpectralMemoryGraph
        Knowledge graph to manage.
    analyzer : optional
        Pre-built :class:`~smgp.core.topology.TopologicalAnalyzer`.
    policy : PrunePolicy or None
        If provided, :meth:`prune` will use this policy to select
        nodes.  When *None*, the parent class behaviour is used.
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        analyzer=None,
        policy: PrunePolicy | None = None,
    ) -> None:
        super().__init__(graph, analyzer=analyzer)
        self.policy = policy

    def prune(
        self,
        strategy: str = "low_persistence",
        threshold: float = 0.1,
        count: int = 5,
    ) -> SpectralMemoryGraph:
        """Prune the knowledge graph using the attached policy.

        If :attr:`policy` is set, the policy's
        :meth:`~smgp.memory.prune_policies.PrunePolicy.select_nodes_to_prune`
        method is used to decide which nodes to remove.  Otherwise the
        parent class pruning logic is invoked unchanged.

        Parameters
        ----------
        strategy : str
            Ignored when a policy is set (kept for API compatibility).
        threshold : float
            Ignored when a policy is set.
        count : int
            Number of nodes the policy should attempt to prune.

        Returns
        -------
        SpectralMemoryGraph
            A (new) graph with the selected nodes removed.
        """
        if self.policy is not None:
            return self._prune_with_policy(count)
        # Fall back to parent implementation
        return super().prune(strategy=strategy, threshold=threshold)

    def _prune_with_policy(self, count: int) -> SpectralMemoryGraph:
        """Remove nodes selected by the attached pruning policy.

        Parameters
        ----------
        count : int
            Target number of nodes to prune.

        Returns
        -------
        SpectralMemoryGraph
            New graph with selected nodes removed.
        """
        to_prune = self.policy.select_nodes_to_prune(self.graph, count)  # type: ignore[union-attr]
        prune_set = set(to_prune)
        original_count = self.graph.num_nodes

        # Build a new graph without the pruned nodes
        hd_dim = self.graph.hd.dim
        pruned = SpectralMemoryGraph(hd_dim=hd_dim)

        for nid in self.graph.nodes():
            if nid in prune_set:
                continue
            data = self.graph.get_node(nid)
            if data is not None:
                pruned.add_node(
                    nid,
                    label=data["label"],
                    properties=data["properties"],
                    vector=data["vector"],
                )

        for src, tgt, key, edata in self.graph.edges():
            if src in prune_set or tgt in prune_set:
                continue
            pruned.add_edge(
                src,
                tgt,
                edata.get("relation", ""),
                properties=edata.get("properties", {}),
            )

        removed = original_count - pruned.num_nodes
        self.history.append({
            "strategy": "policy",
            "policy": type(self.policy).__name__,
            "count_requested": count,
            "nodes_before": original_count,
            "nodes_after": pruned.num_nodes,
            "nodes_removed": removed,
        })

        return pruned
