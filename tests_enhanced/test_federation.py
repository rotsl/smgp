"""Tests for :class:`GraphFederator` with an in-memory SQLite source.

Uses **sqlalchemy** to create an ephemeral SQLite database, populates it
with test data, then verifies that federated queries insert the expected
nodes into the local graph.
"""
from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from smgp.core.graph import SpectralMemoryGraph  # noqa: E402
from smgp.integration.federation import GraphFederator  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sqlite_db(tmp_path):
    """Create a file-based SQLite database via sqlalchemy and return the URI."""
    from sqlalchemy import Column, MetaData, String, Table, create_engine

    db_path = tmp_path / "test_fed.db"
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri)
    metadata = MetaData()
    table = Table(
        "entities",
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String),
        Column("value", String),
    )
    metadata.create_all(engine)

    with engine.connect() as conn:
        conn.execute(
            table.insert(),
            [
                {"id": "ent_1", "name": "Alice", "value": "teacher"},
                {"id": "ent_2", "name": "Bob", "value": "student"},
                {"id": "ent_3", "name": "Carol", "value": "researcher"},
            ],
        )
        conn.commit()

    yield uri


@pytest.fixture()
def graph() -> SpectralMemoryGraph:
    return SpectralMemoryGraph(hd_dim=100, seed=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphFederator:
    """Tests for the federator with SQLite backend."""

    def test_add_source(self, graph: SpectralMemoryGraph, sqlite_db: str) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "test_db", "uri": sqlite_db})

        assert "test_db" in fed.list_sources()

    def test_virtual_query_returns_node(
        self, graph: SpectralMemoryGraph, sqlite_db: str
    ) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "test_db", "uri": sqlite_db})

        result = fed.virtual_query("ent_1", source_name="test_db")

        assert "label" in result
        assert "properties" in result

    def test_virtual_query_inserts_node(
        self, graph: SpectralMemoryGraph, sqlite_db: str
    ) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "test_db", "uri": sqlite_db})

        fed.virtual_query("ent_2", source_name="test_db")

        # Node should have been inserted into the graph
        assert graph.num_nodes >= 1

    def test_virtual_query_missing_entity(
        self, graph: SpectralMemoryGraph, sqlite_db: str
    ) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "test_db", "uri": sqlite_db})

        result = fed.virtual_query("nonexistent", source_name="test_db")
        assert result["label"] == "nonexistent"
        assert result["properties"] == {}

    def test_unknown_source_raises(self, graph: SpectralMemoryGraph) -> None:
        fed = GraphFederator(graph)
        with pytest.raises(ValueError, match="Unknown source"):
            fed.virtual_query("ent_1", source_name="missing")

    def test_no_sources_raises(self, graph: SpectralMemoryGraph) -> None:
        fed = GraphFederator(graph)
        with pytest.raises(ValueError, match="No external sources"):
            fed.virtual_query("ent_1")

    def test_clear_ephemeral(
        self, graph: SpectralMemoryGraph, sqlite_db: str
    ) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "test_db", "uri": sqlite_db})

        fed.virtual_query("ent_1", source_name="test_db")
        fed.virtual_query("ent_2", source_name="test_db")

        assert graph.num_nodes >= 1

        fed.clear_ephemeral()
        assert graph.num_nodes == 0

    def test_sparql_not_implemented(self, graph: SpectralMemoryGraph) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "sparql_ds", "uri": "sparql://example.org"})

        with pytest.raises(NotImplementedError, match="SPARQL"):
            fed.virtual_query("entity_x", source_name="sparql_ds")

    def test_neo4j_not_implemented(self, graph: SpectralMemoryGraph) -> None:
        fed = GraphFederator(graph)
        fed.add_external_source({"name": "neo4j_ds", "uri": "neo4j://localhost:7687"})

        with pytest.raises(NotImplementedError, match="Neo4j"):
            fed.virtual_query("entity_y", source_name="neo4j_ds")
