"""FastAPI-based REST API for serving SMGP.

Exposes the SMGP system as a lightweight REST service with endpoints for:
  - Adding knowledge to the graph
  - Querying the knowledge graph
  - Reasoning with the planner
  - Memory management (pruning)
  - System status

References:
  - FastAPI documentation: https://fastapi.tiangolo.com/
"""
from __future__ import annotations

import logging
from typing import Any

from smgp.core.graph import SpectralMemoryGraph
from smgp.memory.lifecycle import MemoryLifecycle
from smgp.reasoning.verifier import ClaimVerifier

logger = logging.getLogger(__name__)

# Global state
_graph: SpectralMemoryGraph | None = None
_verifier: ClaimVerifier | None = None
_lifecycle: MemoryLifecycle | None = None


def _get_graph() -> SpectralMemoryGraph:
    """Get or create the global knowledge graph."""
    global _graph, _verifier, _lifecycle
    if _graph is None:
        _graph = SpectralMemoryGraph(hd_dim=10000, seed=42)
        _verifier = ClaimVerifier(_graph)
        _lifecycle = MemoryLifecycle(_graph)
    return _graph


def _get_verifier() -> ClaimVerifier:
    """Get or create the global verifier."""
    _get_graph()
    return _verifier  # type: ignore


def _get_lifecycle() -> MemoryLifecycle:
    """Get or create the global lifecycle manager."""
    _get_graph()
    return _lifecycle  # type: ignore


# --- Pydantic models for request/response schemas ---
# These MUST be pydantic BaseModel subclasses so FastAPI can parse
# JSON request bodies and generate an OpenAPI schema.

def _make_models():
    """Import pydantic and return request model classes.

    Deferred so the module can be imported without pydantic (non-server use).
    """
    try:
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "pydantic is required for the API server. "
            "Install with: pip install smgp[integration]"
        ) from exc

    class AddKnowledgeRequest(BaseModel):
        """Request body for /add_knowledge."""
        subject: str
        relation: str
        obj: str
        properties: dict[str, Any] | None = None

    class QueryRequest(BaseModel):
        """Request body for /query."""
        query: str
        k: int = 5
        node_type: str | None = None

    class ReasonRequest(BaseModel):
        """Request body for /reason."""
        claim: str

    class PruneRequest(BaseModel):
        """Request body for /memory/prune."""
        strategy: str = "low_persistence"
        threshold: float = 0.1

    return AddKnowledgeRequest, QueryRequest, ReasonRequest, PruneRequest


# --- Endpoint handlers ---

def add_knowledge(request: Any) -> dict[str, Any]:
    """Add knowledge to the graph."""
    graph = _get_graph()

    if request.subject not in graph.graph:
        graph.add_node(
            request.subject,
            label="entity",
            properties={"name": request.subject},
        )
    if request.obj not in graph.graph:
        graph.add_node(
            request.obj,
            label="entity",
            properties={"name": request.obj},
        )

    edge_id = graph.add_edge(request.subject, request.obj, request.relation)

    return {
        "status": "ok",
        "edge_id": edge_id,
        "subject": request.subject,
        "relation": request.relation,
        "object": request.obj,
    }


def query(request: Any) -> dict[str, Any]:
    """Query the knowledge graph by HD vector similarity."""
    import numpy as np

    graph = _get_graph()
    hd = graph.hd
    seed_val = hash(request.query) & 0xFFFFFFFF
    rng = np.random.default_rng(seed_val)
    query_vec = rng.choice(np.array([-1, 1], dtype=np.int8), size=hd.dim)

    results = graph.query_similar(query_vec, k=request.k, node_type=request.node_type)

    return {
        "status": "ok",
        "query": request.query,
        "results": [
            {"node_id": nid, "similarity": float(sim)}
            for nid, sim in results
        ],
    }


def reason(request: Any) -> dict[str, Any]:
    """Verify a claim against the knowledge graph."""
    verifier = _get_verifier()
    result = verifier.verify(request.claim)
    return {"status": "ok", **result}


def prune_memory(request: Any) -> dict[str, Any]:
    """Prune the knowledge graph memory."""
    global _graph, _verifier, _lifecycle
    lifecycle = _get_lifecycle()
    graph = _get_graph()

    nodes_before = graph.num_nodes
    edges_before = graph.num_edges

    pruned = lifecycle.prune(strategy=request.strategy, threshold=request.threshold)

    _graph = pruned
    _verifier = ClaimVerifier(pruned)
    _lifecycle = MemoryLifecycle(pruned)

    return {
        "status": "ok",
        "nodes_before": nodes_before,
        "nodes_after": pruned.num_nodes,
        "edges_before": edges_before,
        "edges_after": pruned.num_edges,
        "nodes_removed": nodes_before - pruned.num_nodes,
    }


def status() -> dict[str, Any]:
    """Get system status."""
    graph = _get_graph()
    lifecycle = _get_lifecycle()
    evaluation = lifecycle.evaluate()

    return {
        "status": "ok",
        "graph": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "hd_dim": graph.hd.dim,
        },
        "memory": evaluation,
    }


# --- FastAPI app factory ---

def create_app() -> Any:
    """Create the FastAPI application.

    Returns:
        FastAPI app instance with all routes registered.

    Raises:
        ImportError: if fastapi or pydantic are not installed.
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        logger.warning("FastAPI not installed. API module available for import only.")
        return None

    # Build typed request models — raises ImportError early if pydantic missing
    AddKnowledgeRequest, QueryRequest, ReasonRequest, PruneRequest = _make_models()

    app = FastAPI(
        title="SMGP API",
        description="Spectral Memory Graph Processor REST API",
        version="1.0.0",
    )

    # Wrap generic handlers with typed signatures FastAPI/Pydantic v2 can introspect
    async def _add_knowledge(request: AddKnowledgeRequest):  # type: ignore[valid-type]
        return add_knowledge(request)

    async def _query(request: QueryRequest):  # type: ignore[valid-type]
        return query(request)

    async def _reason(request: ReasonRequest):  # type: ignore[valid-type]
        return reason(request)

    async def _prune_memory(request: PruneRequest):  # type: ignore[valid-type]
        return prune_memory(request)

    app.add_api_route("/add_knowledge", _add_knowledge, methods=["POST"])
    app.add_api_route("/query", _query, methods=["POST"])
    app.add_api_route("/reason", _reason, methods=["POST"])
    app.add_api_route("/memory/prune", _prune_memory, methods=["POST"])
    app.add_api_route("/status", status, methods=["GET"])

    @app.get("/")
    def root():
        return {"name": "SMGP", "version": "1.0.0", "docs": "/docs"}

    return app
