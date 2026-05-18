"""Federated knowledge-base access for SMGP graphs.

:class:`GraphFederator` allows a local :class:`~smgp.core.graph.SpectralMemoryGraph`
to lazily pull in entity data from external sources (SQLite, SPARQL, Neo4j).
Fetched entities are inserted as **ephemeral** nodes so the local graph
can answer queries that span both local and remote knowledge.

Only SQLite is fully implemented; SPARQL and Neo4j connectors are provided
as placeholders that raise :class:`NotImplementedError`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smgp.core.graph import SpectralMemoryGraph


class GraphFederator:
    """Lazily federates external knowledge bases into a local SMGP graph.

    Parameters
    ----------
    local_graph : SpectralMemoryGraph
        The local graph that receives ephemeral nodes.
    """

    def __init__(self, local_graph: SpectralMemoryGraph) -> None:

        self.local_graph: SpectralMemoryGraph = local_graph
        self._sources: dict[str, dict[str, Any]] = {}
        self._ephemeral_nodes: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_external_source(self, source_config: dict[str, Any]) -> None:
        """Register an external data source.

        Parameters
        ----------
        source_config : dict
            Must contain:
              - ``"uri"``  — connection string, e.g.
                ``"sqlite:///path/to/db"``, ``"sparql://..."``, ``"neo4j://..."``.
              - ``"name"`` — human-readable identifier used as the key for
                :meth:`virtual_query`.
        """
        name = source_config["name"]
        self._sources[name] = source_config

    def virtual_query(
        self,
        entity_id: str,
        source_name: str | None = None,
    ) -> dict[str, Any]:
        """Fetch entity data from an external source and insert an ephemeral node.

        Parameters
        ----------
        entity_id : str
            Identifier used to look up the entity in the external source.
        source_name : str or None
            Name of a registered source.  If *None* the first registered
            source is used.

        Returns
        -------
        dict
            Node data dict with keys ``"label"``, ``"properties"``, ``"vector"``.
            Also contains ``"source"`` and ``"entity_id"`` metadata.

        Raises
        ------
        ValueError
            If *source_name* is provided but not registered.
        NotImplementedError
            If the source URI scheme has no fetch implementation.
        """
        if source_name is not None and source_name not in self._sources:
            raise ValueError(
                f"Unknown source '{source_name}'. "
                f"Available: {list(self._sources.keys())}"
            )

        # Pick the source
        if source_name is not None:
            config = self._sources[source_name]
        else:
            if not self._sources:
                raise ValueError("No external sources registered.")
            config = next(iter(self._sources.values()))

        uri: str = config["uri"]

        if uri.startswith("sqlite:///") or uri.startswith("sqlite://"):
            node_data = self._fetch_sqlite(config, entity_id)
        elif uri.startswith("sparql://"):
            node_data = self._fetch_sparql(config, entity_id)
        elif uri.startswith("neo4j://"):
            node_data = self._fetch_neo4j(config, entity_id)
        else:
            raise NotImplementedError(
                f"URI scheme '{uri.split(':')[0]}://' is not supported"
            )

        # Insert as an ephemeral node in the local graph
        ephemeral_id = f"ephemeral:{source_name or 'default'}:{entity_id}"
        self.local_graph.add_node(
            ephemeral_id,
            label=node_data.get("label", entity_id),
            properties=node_data.get("properties", {}),
            vector=node_data.get("vector"),
        )
        self._ephemeral_nodes.add(ephemeral_id)

        return {
            "label": node_data.get("label", entity_id),
            "properties": node_data.get("properties", {}),
            "vector": node_data.get("vector"),
            "source": source_name or next(iter(self._sources.keys())),
            "entity_id": entity_id,
        }

    # ------------------------------------------------------------------
    # Source-specific fetchers
    # ------------------------------------------------------------------

    def _fetch_sqlite(
        self,
        source_config: dict[str, Any],
        entity_id: str,
    ) -> dict[str, Any]:
        """Fetch entity data from a SQLite database.

        Uses **sqlalchemy** (lazy-imported) to execute a parameterised query.

        The SQLAlchemy engine is created from the ``"uri"`` key in
        *source_config*.

        Parameters
        ----------
        source_config : dict
            Must contain ``"uri"`` (e.g. ``"sqlite:///mydb.sqlite"``).
        entity_id : str
            The primary-key value to look up.

        Returns
        -------
        dict
            ``{"label": ..., "properties": {column: value, ...}}``
        """
        try:
            from sqlalchemy import create_engine, text  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for SQLite federation.  "
                "Install it with: pip install sqlalchemy"
            ) from exc

        uri: str = source_config["uri"]
        engine = create_engine(uri)

        # Attempt a generic lookup: assume a table called 'entities' with
        # columns (id, name, value).  The caller can override via config.
        table_name = source_config.get("table", "entities")
        id_column = source_config.get("id_column", "id")

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT * FROM {table_name} WHERE {id_column} = :eid"
                ),
                {"eid": entity_id},
            ).fetchone()

        if row is None:
            return {"label": entity_id, "properties": {}}

        # Build properties from the row
        columns = source_config.get(
            "columns",
            [id_column, "name", "value"],
        )
        properties: dict[str, Any] = {}
        for col_name, val in zip(columns, row):
            properties[col_name] = val

        label = properties.pop("name", entity_id)

        return {"label": label, "properties": properties}

    @staticmethod
    def _fetch_sparql(
        source_config: dict[str, Any],
        entity_id: str,
    ) -> dict[str, Any]:
        """Placeholder for SPARQL endpoint queries.

        Raises :class:`NotImplementedError` until a SPARQL library is
        integrated.
        """
        raise NotImplementedError(
            "SPARQL federation is not yet implemented.  "
            "Contributions welcome."
        )

    @staticmethod
    def _fetch_neo4j(
        source_config: dict[str, Any],
        entity_id: str,
    ) -> dict[str, Any]:
        """Placeholder for Neo4j queries.

        Raises :class:`NotImplementedError` until the Neo4j driver is
        integrated.
        """
        raise NotImplementedError(
            "Neo4j federation is not yet implemented.  "
            "Contributions welcome."
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        """Return the names of all registered external sources."""
        return list(self._sources.keys())

    def clear_ephemeral(self) -> None:
        """Remove all ephemeral nodes from the local graph."""
        for nid in list(self._ephemeral_nodes):
            self.local_graph.remove_node(nid)
        self._ephemeral_nodes.clear()
