"""Pluggable pruning policies for the memory lifecycle manager.

Each policy implements :class:`PrunePolicy` and decides *which* nodes to
remove from a :class:`~smgp.core.graph.SpectralMemoryGraph`.  The
:class:`~smgp.memory.enhanced_lifecycle.EnhancedMemoryLifecycle` class
accepts any policy instance to customise its pruning behaviour.

Built-in policies:

* :class:`LRUPrunePolicy` — evict least-recently-accessed nodes.
* :class:`TimeBasedPrunePolicy` — evict nodes whose last access exceeds a TTL.
* :class:`AccessCountPrunePolicy` — evict nodes with too few accesses.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smgp.core.graph import SpectralMemoryGraph


class PrunePolicy(ABC):
    """Abstract base class for node-pruning strategies.

    Subclasses must implement :meth:`select_nodes_to_prune`, which
    returns up to *count* node IDs that should be removed from the graph.
    """

    @abstractmethod
    def select_nodes_to_prune(
        self,
        graph: SpectralMemoryGraph,
        count: int,
    ) -> list[str]:
        """Return *count* node IDs to prune from *graph*.

        Parameters
        ----------
        graph : SpectralMemoryGraph
            The graph whose nodes are being considered for removal.
        count : int
            Maximum number of nodes to select.  The returned list may be
            shorter if fewer nodes are eligible.

        Returns
        -------
        list[str]
            Node IDs to prune, ordered by priority (first removed first).
        """


class LRUPrunePolicy(PrunePolicy):
    """Least-Recently-Used pruning policy.

    Maintains a per-node access timestamp (monotonic wall-clock time) and
    evicts the nodes that were accessed longest ago.
    """

    def __init__(self) -> None:
        self._access_times: dict[str, float] = {}

    def touch(self, node_id: str) -> None:
        """Record that *node_id* was accessed at the current time."""
        self._access_times[node_id] = time.monotonic()

    def select_nodes_to_prune(
        self,
        graph: SpectralMemoryGraph,
        count: int,
    ) -> list[str]:
        """Return the *count* least-recently-accessed nodes.

        Nodes with no recorded access time are treated as oldest.
        """
        node_ids = list(graph.nodes())
        # Sort by access time ascending (oldest first); unrecorded → 0
        node_ids.sort(key=lambda nid: self._access_times.get(nid, 0.0))
        return node_ids[:count]


class TimeBasedPrunePolicy(PrunePolicy):
    """Time-To-Live pruning policy.

    Nodes whose last access time is more than *ttl_seconds* ago are
    eligible for pruning.
    """

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._access_times: dict[str, float] = {}

    def touch(self, node_id: str) -> None:
        """Record that *node_id* was accessed at the current time."""
        self._access_times[node_id] = time.monotonic()

    def select_nodes_to_prune(
        self,
        graph: SpectralMemoryGraph,
        count: int,
    ) -> list[str]:
        """Return up to *count* nodes whose last access exceeds the TTL.

        If fewer than *count* nodes have expired, all expired nodes are
        returned.
        """
        now = time.monotonic()
        expired: list[str] = []
        for nid in graph.nodes():
            last_access = self._access_times.get(nid, 0.0)
            if now - last_access > self.ttl_seconds:
                expired.append(nid)
        # Return most-stale first
        expired.sort(key=lambda nid: self._access_times.get(nid, 0.0))
        return expired[:count]


class AccessCountPrunePolicy(PrunePolicy):
    """Access-count pruning policy.

    Nodes that have been accessed fewer than *min_accesses* times are
    eligible for pruning.
    """

    def __init__(self, min_accesses: int = 1) -> None:
        self.min_accesses = min_accesses
        self._access_counts: dict[str, int] = {}

    def touch(self, node_id: str) -> None:
        """Increment the access counter for *node_id*."""
        self._access_counts[node_id] = self._access_counts.get(node_id, 0) + 1

    def select_nodes_to_prune(
        self,
        graph: SpectralMemoryGraph,
        count: int,
    ) -> list[str]:
        """Return up to *count* nodes with access count below the threshold.

        Nodes are ordered by ascending access count so the least-accessed
        nodes are pruned first.
        """
        below: list[str] = []
        for nid in graph.nodes():
            accesses = self._access_counts.get(nid, 0)
            if accesses < self.min_accesses:
                below.append(nid)
        # Sort by ascending count (least accessed first)
        below.sort(key=lambda nid: self._access_counts.get(nid, 0))
        return below[:count]
