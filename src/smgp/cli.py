"""Command-line interface for the Spectral Memory Graph Processor (SMGP).

Provides commands for graph management, claim verification, server
launching, and version inspection. All commands load configuration
from environment variables via SMGPConfig.from_env().

References:
    Angles, R. & Gutierrez, C. (2008). "Survey of Graph Database Models."
        ACM Computing Surveys, 40(1), 1-39.
"""
from __future__ import annotations

import json
import sys

import click

import smgp
from smgp.config import SMGPConfig
from smgp.core.graph import SpectralMemoryGraph
from smgp.reasoning.verifier import ClaimVerifier


@click.group()
@click.version_option(version=smgp.__version__, prog_name="smgp")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to a YAML or JSON configuration file.",
)
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """SMGP — Spectral Memory Graph Processor.

    Persistent, hallucination-free AI reasoning through spectral
    memory graphs and hyperdimensional computing.
    """
    ctx.ensure_object(dict)
    if config:
        ctx.obj["config"] = SMGPConfig.from_yaml(config)
    else:
        ctx.obj["config"] = SMGPConfig.from_env()


# ---------------------------------------------------------------------------
# graph sub-group
# ---------------------------------------------------------------------------

@cli.group()
def graph() -> None:
    """Graph management commands."""


@graph.command("stats")
@click.option(
    "--hd-dim",
    type=int,
    default=None,
    help="HD vector dimensionality (overrides config).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "--load",
    "load_path",
    type=click.Path(exists=True),
    default=None,
    help="Load graph from a JSON file instead of creating a new one.",
)
@click.pass_context
def graph_stats(
    ctx: click.Context,
    hd_dim: int | None,
    seed: int | None,
    load_path: str | None,
) -> None:
    """Print graph statistics (num_nodes, num_edges, etc.)."""
    cfg: SMGPConfig = ctx.obj["config"]
    effective_hd_dim = hd_dim if hd_dim is not None else cfg.hd_dim
    effective_seed = seed if seed is not None else cfg.seed

    if load_path:
        g = _load_graph_from_file(load_path)
    else:
        g = SpectralMemoryGraph(
            hd_dim=effective_hd_dim,
            seed=effective_seed,
        )

    click.echo(f"Nodes:   {g.num_nodes}")
    click.echo(f"Edges:   {g.num_edges}")
    click.echo(f"HD Dim:  {g.hd.dim}")
    click.echo(f"Isolated nodes: {_count_isolated(g)}")


@graph.command("query")
@click.argument("node_id", type=str)
@click.option(
    "-k",
    "--top-k",
    type=int,
    default=5,
    help="Number of similar nodes to return.",
)
@click.option(
    "--load",
    "load_path",
    type=click.Path(exists=True),
    required=True,
    help="Load graph from a JSON file.",
)
@click.pass_context
def graph_query(
    ctx: click.Context,
    node_id: str,
    top_k: int,
    load_path: str,
) -> None:
    """Query similar nodes to the given NODE_ID."""
    g = _load_graph_from_file(load_path)

    node_data = g.get_node(node_id)
    if node_data is None:
        click.echo(f"Error: node '{node_id}' not found in graph.", err=True)
        sys.exit(1)

    query_vec = node_data["vector"]
    results = g.query_similar(query_vec, k=top_k)

    click.echo(f"Top-{top_k} nodes similar to '{node_id}':")
    for rank, (nid, sim) in enumerate(results, 1):
        label = g.get_node(nid)
        lbl_str = label["label"] if label else ""
        click.echo(f"  {rank}. {nid}  (similarity={sim:.4f}, label={lbl_str})")


# ---------------------------------------------------------------------------
# verify (top-level command)
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("claim", type=str)
@click.option(
    "--load",
    "load_path",
    type=click.Path(exists=True),
    default=None,
    help="Load graph from a JSON file for verification.",
)
@click.pass_context
def verify(ctx: click.Context, claim: str, load_path: str | None) -> None:
    """Verify a factual CLAIM against the knowledge graph."""
    if load_path:
        g = _load_graph_from_file(load_path)
    else:
        g = SpectralMemoryGraph(
            hd_dim=ctx.obj["hd_dim"] if "hd_dim" in ctx.obj else 10000,
            seed=ctx.obj.get("seed"),
        )

    verifier = ClaimVerifier(g)
    result = verifier.verify(claim)

    status = "VERIFIED" if result["verified"] else "NOT VERIFIED"
    click.echo(f"Claim:   {claim}")
    click.echo(f"Status:  {status}")
    click.echo(f"Confidence: {result['confidence']:.2f}")


# ---------------------------------------------------------------------------
# server (top-level command)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to bind the server to.",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for development.",
)
def server(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server (requires smgp[integration])."""
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Error: uvicorn is required. Install with: pip install smgp[integration]",
            err=True,
        )
        sys.exit(1)

    try:
        from smgp.integration.api import create_app
    except ImportError:
        click.echo(
            "Error: FastAPI integration not found. Install with: pip install smgp[integration]",
            err=True,
        )
        sys.exit(1)

    app = create_app()
    click.echo(f"Starting SMGP server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# version (top-level command)
# ---------------------------------------------------------------------------


@cli.command()
def version() -> None:
    """Print the SMGP version."""
    click.echo(smgp.__version__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _load_graph_from_file(path: str) -> SpectralMemoryGraph:
    """Load a SpectralMemoryGraph from a JSON file.

    Expected JSON format:
    {
        "hd_dim": 10000,
        "seed": 42,
        "nodes": {"id": {"label": "...", "properties": {}}},
        "edges": [{"source": "...", "target": "...", "relation": "..."}]
    }

    Args:
        path: Path to the JSON file.

    Returns:
        Reconstructed SpectralMemoryGraph.
    """
    with open(path) as f:
        data = json.load(f)

    hd_dim = data.get("hd_dim", 10000)
    seed = data.get("seed", None)
    g = SpectralMemoryGraph(hd_dim=hd_dim, seed=seed)

    for node_id, node_data in data.get("nodes", {}).items():
        g.add_node(
            node_id,
            label=node_data.get("label", ""),
            properties=node_data.get("properties", {}),
        )

    for edge_data in data.get("edges", []):
        g.add_edge(
            edge_data["source"],
            edge_data["target"],
            edge_data.get("relation", "related"),
            properties=edge_data.get("properties", {}),
        )

    return g


def _count_isolated(g: SpectralMemoryGraph) -> int:
    """Count nodes with zero in-degree and out-degree."""
    count = 0
    for nid in g.nodes():
        if not g.neighbors(nid) and not g.predecessors(nid):
            count += 1
    return count


if __name__ == "__main__":
    cli()
