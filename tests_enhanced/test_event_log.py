"""Tests for Event-Sourced History Log."""
from __future__ import annotations

import pytest


class TestEventLog:
    """Tests for the EventLog class."""

    def test_import(self):
        """Module should be importable."""
        from smgp.memory.event_log import EventLog
        assert EventLog is not None

    def test_jsonl_backend(self, tmp_path):
        """JSONL backend should append and read events."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "test.jsonl")
        log = EventLog(backend="jsonl", path=path)

        log.append({"type": "add_node", "node_id": "n1"})
        log.append({"type": "add_edge", "source": "n1", "target": "n2"})

        events = log.read_all()
        assert len(events) == 2
        assert events[0]["type"] == "add_node"
        assert events[1]["type"] == "add_edge"

    def test_jsonl_timestamp_auto_added(self, tmp_path):
        """Events should get timestamp automatically if not provided."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "ts.jsonl")
        log = EventLog(backend="jsonl", path=path)

        log.append({"type": "test", "value": 42})
        events = log.read_all()
        assert "timestamp" in events[0]
        assert events[0]["value"] == 42

    def test_jsonl_clear(self, tmp_path):
        """Clear should remove all events."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "clear.jsonl")
        log = EventLog(backend="jsonl", path=path)

        log.append({"type": "test"})
        assert len(log) == 1

        log.clear()
        assert len(log) == 0

    def test_jsonl_empty_read(self, tmp_path):
        """Reading from non-existent log should return empty list."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "nonexistent.jsonl")
        log = EventLog(backend="jsonl", path=path)
        assert log.read_all() == []

    def test_sqlite_backend(self, tmp_path):
        """SQLite backend should append and read events."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "test.db")
        log = EventLog(backend="sqlite", path=path)

        log.append({"type": "add_node", "node_id": "s1"})
        log.append({"type": "add_edge", "source": "s1", "target": "s2"})

        events = log.read_all()
        assert len(events) == 2
        assert events[0]["node_id"] == "s1"

    def test_sqlite_clear(self, tmp_path):
        """SQLite clear should remove all events."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "clear.db")
        log = EventLog(backend="sqlite", path=path)

        log.append({"type": "test"})
        log.append({"type": "test2"})
        assert len(log) == 2

        log.clear()
        assert len(log) == 0

    def test_unsupported_backend_raises(self):
        """Unsupported backend should raise ValueError."""
        from smgp.memory.event_log import EventLog
        with pytest.raises(ValueError, match="Unsupported backend"):
            EventLog(backend="mongodb")

    def test_len(self, tmp_path):
        """len(log) should return number of events."""
        from smgp.memory.event_log import EventLog

        path = str(tmp_path / "len.jsonl")
        log = EventLog(backend="jsonl", path=path)
        assert len(log) == 0

        for i in range(5):
            log.append({"type": "test", "i": i})
        assert len(log) == 5


class TestEventSourcedGraph:
    """Tests for the EventSourcedGraph class."""

    def test_import(self):
        """Module should be importable."""
        from smgp.memory.event_log import EventSourcedGraph
        assert EventSourcedGraph is not None

    def test_add_node_logs_event(self, tmp_path):
        """Adding a node should log an event."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "esg_node.jsonl")
        log = EventLog(backend="jsonl", path=path)
        esg = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)

        esg.add_node("n1", label="entity", properties={"name": "Alice"})

        events = log.read_all()
        assert len(events) == 1
        assert events[0]["type"] == "add_node"
        assert events[0]["node_id"] == "n1"
        assert events[0]["label"] == "entity"
        assert events[0]["properties"]["name"] == "Alice"
        assert events[0]["vector"] is not None

    def test_add_edge_logs_event(self, tmp_path):
        """Adding an edge should log an event."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "esg_edge.jsonl")
        log = EventLog(backend="jsonl", path=path)
        esg = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)

        esg.add_node("a", label="x")
        esg.add_node("b", label="y")
        esg.add_edge("a", "b", "related")

        events = log.read_all()
        # 2 add_node events + 1 add_edge event
        assert len(events) == 3
        assert events[2]["type"] == "add_edge"
        assert events[2]["source_id"] == "a"
        assert events[2]["target_id"] == "b"
        assert events[2]["relation"] == "related"

    def test_remove_node_logs_event(self, tmp_path):
        """Removing a node should log an event."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "esg_rm.jsonl")
        log = EventLog(backend="jsonl", path=path)
        esg = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)

        esg.add_node("to_remove")
        esg.remove_node("to_remove")

        events = log.read_all()
        assert len(events) == 2
        assert events[1]["type"] == "remove_node"
        assert events[1]["node_id"] == "to_remove"

    def test_replay_from_identical_state(self, tmp_path):
        """Replaying from log should produce identical graph state."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "replay.jsonl")
        log = EventLog(backend="jsonl", path=path)

        # Build original graph with mutations
        original = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)
        original.add_node("alice", label="person", properties={"age": 30})
        original.add_node("bob", label="person", properties={"age": 25})
        original.add_node("math", label="concept")
        original.add_edge("alice", "bob", "knows")
        original.add_edge("alice", "math", "studies")

        original_nodes = set(original.nodes())
        original_edges = set(
            (e[0], e[1], e[3].get("relation", "")) for e in original.edges()
        )

        # Replay from log into a new graph
        replayed = EventSourcedGraph.replay_from(log)

        replayed_nodes = set(replayed.nodes())
        replayed_edges = set(
            (e[0], e[1], e[3].get("relation", "")) for e in replayed.edges()
        )

        assert replayed_nodes == original_nodes
        assert replayed_edges == original_edges
        assert replayed.num_nodes == original.num_nodes
        assert replayed.num_edges == original.num_edges

    def test_replay_with_removals(self, tmp_path):
        """Replay with removals should produce correct final state."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "replay_rm.jsonl")
        log = EventLog(backend="jsonl", path=path)

        esg = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)
        esg.add_node("keep", label="x")
        esg.add_node("remove_me", label="y")
        esg.add_edge("keep", "remove_me", "link")
        esg.remove_node("remove_me")

        # After replay, only "keep" should exist
        replayed = EventSourcedGraph.replay_from(log)
        assert "keep" in replayed.nodes()
        assert "remove_me" not in replayed.nodes()
        assert replayed.num_nodes == 1
        assert replayed.num_edges == 0

    def test_replay_empty_log(self, tmp_path):
        """Replaying from empty log should produce empty graph."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "empty.jsonl")
        log = EventLog(backend="jsonl", path=path)
        esg = EventSourcedGraph.replay_from(log)
        assert esg.num_nodes == 0
        assert esg.num_edges == 0

    def test_sqlite_replay(self, tmp_path):
        """Replay should work with SQLite backend."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "replay.db")
        log = EventLog(backend="sqlite", path=path)

        original = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)
        original.add_node("sql_a", label="test")
        original.add_node("sql_b", label="test2")
        original.add_edge("sql_a", "sql_b", "rel")

        original_nodes = set(original.nodes())
        replayed = EventSourcedGraph.replay_from(log)
        assert set(replayed.nodes()) == original_nodes
        assert replayed.num_edges == 1

    def test_repr(self, tmp_path):
        """repr should be informative."""
        from smgp.memory.event_log import EventLog, EventSourcedGraph

        path = str(tmp_path / "repr.jsonl")
        log = EventLog(backend="jsonl", path=path)
        esg = EventSourcedGraph(hd_dim=1000, seed=42, event_log=log)
        esg.add_node("x")
        r = repr(esg)
        assert "EventSourcedGraph" in r
        assert "nodes=1" in r
