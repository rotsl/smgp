"""Event-Sourced History Log for SMGP.

Provides EventLog (JSONL or SQLite-backed) and EventSourcedGraph, a subclass
of SpectralMemoryGraph that records every mutation as an event, enabling full
state reconstruction via replay.

References:
  - Event Sourcing: Fowler, M. (2005). "Event Sourcing."
    https://martinfowler.com/eaaDev/EventSourcing.html
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("jsonl", "sqlite")


def _serialize_vector(vector: np.ndarray | None) -> list[int] | None:
    """Serialize a numpy vector to a JSON-compatible list.

    Args:
        vector: Numpy array or None.

    Returns:
        List of ints or None.
    """
    if vector is None:
        return None
    return vector.tolist()


def _deserialize_vector(data: list[int] | None) -> np.ndarray | None:
    """Deserialize a list back to a numpy vector.

    Args:
        data: List of ints or None.

    Returns:
        Numpy array or None.
    """
    if data is None:
        return None
    return np.array(data, dtype=np.int8)


class EventLog:
    """Append-only event log backed by JSONL files or SQLite.

    Records graph mutations as structured events for event sourcing.
    Each event is a dict with keys: type, timestamp, and event-specific data.

    Attributes:
        backend: Storage backend ("jsonl" or "sqlite").
        path: File path for the log storage.
    """

    def __init__(
        self,
        backend: str = "jsonl",
        path: str | None = None,
    ) -> None:
        """Initialize the event log.

        Args:
            backend: Storage backend, either "jsonl" or "sqlite".
            path: File path for the log. If None, uses a default temp path.
        """
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported: {SUPPORTED_BACKENDS}"
            )
        self.backend = backend

        if path is None:
            if backend == "jsonl":
                self.path = os.path.join(
                    os.path.dirname(__file__), "_default_event_log.jsonl"
                )
            else:
                self.path = os.path.join(
                    os.path.dirname(__file__), "_default_event_log.db"
                )
        else:
            self.path = path

        # Ensure parent directory exists
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if self.backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite table for event storage."""
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def append(self, event: dict[str, Any]) -> None:
        """Append an event to the log.

        Args:
            event: Event dict. Should contain 'type' key. 'timestamp' is
                added automatically if not present.
        """
        if "timestamp" not in event:
            event["timestamp"] = time.time()

        if self.backend == "jsonl":
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        elif self.backend == "sqlite":
            conn = sqlite3.connect(self.path)
            try:
                conn.execute(
                    "INSERT INTO events (event_data, timestamp) VALUES (?, ?)",
                    (json.dumps(event, default=str), event["timestamp"]),
                )
                conn.commit()
            finally:
                conn.close()

    def read_all(self) -> list[dict[str, Any]]:
        """Read all events from the log.

        Returns:
            List of event dicts, in order.
        """
        events: list[dict[str, Any]] = []

        if self.backend == "jsonl":
            if not os.path.exists(self.path):
                return events
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        elif self.backend == "sqlite":
            conn = sqlite3.connect(self.path)
            try:
                cursor = conn.execute(
                    "SELECT event_data FROM events ORDER BY id ASC"
                )
                for row in cursor:
                    events.append(json.loads(row[0]))
            finally:
                conn.close()

        return events

    def clear(self) -> None:
        """Clear all events from the log."""
        if self.backend == "jsonl":
            if os.path.exists(self.path):
                os.remove(self.path)
        elif self.backend == "sqlite":
            conn = sqlite3.connect(self.path)
            try:
                conn.execute("DELETE FROM events")
                conn.commit()
            finally:
                conn.close()

    def __len__(self) -> int:
        """Return the number of events in the log."""
        return len(self.read_all())

    def __repr__(self) -> str:
        return f"EventLog(backend={self.backend!r}, path={self.path!r}, events={len(self)})"


class EventSourcedGraph:
    """Event-sourced wrapper around SpectralMemoryGraph.

    Every mutation (add_node, add_edge, remove_node, remove_edge) is logged
    to an EventLog. The graph can be reconstructed from the log via
    ``replay_from``.

    Attributes:
        graph: The underlying SpectralMemoryGraph.
        event_log: The EventLog recording mutations.
    """

    def __init__(
        self,
        hd_dim: int = 10000,
        seed: int | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        """Initialize an event-sourced graph.

        Args:
            hd_dim: Dimensionality of HD vectors.
            seed: Random seed for reproducibility.
            event_log: Optional EventLog. If None, creates an in-memory
                JSONL-backed log.
        """
        from smgp.core.graph import SpectralMemoryGraph

        self.graph = SpectralMemoryGraph(hd_dim=hd_dim, seed=seed)
        if event_log is not None:
            self.event_log = event_log
        else:
            self.event_log = EventLog(backend="jsonl")

    def add_node(
        self,
        node_id: str,
        label: str = "",
        properties: dict[str, Any] | None = None,
        vector: np.ndarray | None = None,
    ) -> str:
        """Add a node and log the event.

        Args:
            node_id: Unique node identifier.
            label: Semantic type label.
            properties: Optional metadata.
            vector: Optional HD vector.

        Returns:
            The node_id.
        """
        result = self.graph.add_node(
            node_id, label=label, properties=properties, vector=vector
        )
        # Retrieve the actual vector that was assigned/generated
        node_data = self.graph.get_node(node_id)
        actual_vector = node_data["vector"] if node_data else None

        self.event_log.append({
            "type": "add_node",
            "node_id": node_id,
            "label": label,
            "properties": properties,
            "vector": _serialize_vector(actual_vector),
        })
        return result

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Add an edge and log the event.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Relation type.
            properties: Optional edge metadata.

        Returns:
            The edge key.
        """
        edge_key = self.graph.add_edge(
            source_id, target_id, relation, properties=properties
        )
        self.event_log.append({
            "type": "add_edge",
            "edge_key": edge_key,
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation,
            "properties": properties,
        })
        return edge_key

    def remove_node(self, node_id: str) -> None:
        """Remove a node and log the event.

        Args:
            node_id: Node to remove.
        """
        self.graph.remove_node(node_id)
        self.event_log.append({
            "type": "remove_node",
            "node_id": node_id,
        })

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge and log the event.

        Args:
            edge_id: Edge key to remove.
        """
        self.graph.remove_edge(edge_id)
        self.event_log.append({
            "type": "remove_edge",
            "edge_id": edge_id,
        })

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Retrieve node data (delegates to inner graph)."""
        return self.graph.get_node(node_id)

    def query_similar(
        self,
        vector: np.ndarray,
        k: int = 5,
        node_type: str | None = None,
    ):
        """Query similar nodes (delegates to inner graph)."""
        return self.graph.query_similar(vector, k=k, node_type=node_type)

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return self.graph.num_nodes

    @property
    def num_edges(self) -> int:
        """Number of edges."""
        return self.graph.num_edges

    def nodes(self):
        """Return list of node IDs."""
        return self.graph.nodes()

    def edges(self):
        """Return list of edges."""
        return self.graph.edges()

    @classmethod
    def replay_from(cls, log: EventLog) -> EventSourcedGraph:
        """Rebuild an EventSourcedGraph by replaying events from a log.

        Processes events in order, recreating the exact graph state.

        Args:
            log: EventLog containing the recorded events.

        Returns:
            A new EventSourcedGraph with the replayed state.
        """
        esg = cls(hd_dim=10000, seed=42, event_log=log)
        events = log.read_all()

        for event in events:
            event_type = event.get("type")

            if event_type == "add_node":
                vector = _deserialize_vector(event.get("vector"))
                esg.graph.add_node(
                    node_id=event["node_id"],
                    label=event.get("label", ""),
                    properties=event.get("properties"),
                    vector=vector,
                )
            elif event_type == "add_edge":
                esg.graph.add_edge(
                    source_id=event["source_id"],
                    target_id=event["target_id"],
                    relation=event["relation"],
                    properties=event.get("properties"),
                )
            elif event_type == "remove_node":
                try:
                    esg.graph.remove_node(event["node_id"])
                except KeyError:
                    pass  # Node may have already been removed
            elif event_type == "remove_edge":
                try:
                    esg.graph.remove_edge(event["edge_id"])
                except KeyError:
                    pass  # Edge may have already been removed
            else:
                logger.warning("Unknown event type: %s", event_type)

        logger.info("Replayed %d events from log", len(events))
        return esg

    def __repr__(self) -> str:
        return (
            f"EventSourcedGraph("
            f"nodes={self.num_nodes}, "
            f"edges={self.num_edges}, "
            f"log_events={len(self.event_log)})"
        )
