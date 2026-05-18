"""I/O utilities for saving and loading SMGP graphs and checkpoints.

Supports serialization of knowledge graphs, including HD vectors, to
NPZ (NumPy) and JSON formats.

References:
  - Angles, R. & Gutierrez, C. (2008). "Graph Database Export Formats."
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smgp.core.graph import SpectralMemoryGraph


def save_graph(
    graph: SpectralMemoryGraph,
    path: str,
    format: str = "npz",
) -> str:
    """Save a knowledge graph to disk.

    Args:
        graph: The SpectralMemoryGraph to save.
        path: Output file path (without extension).
        format: Serialization format — 'npz' or 'json'.

    Returns:
        Path to the saved file.
    """
    save_path = Path(path)

    if format == "npz":
        # Save HD vectors and metadata as NPZ
        node_ids = list(graph.nodes())
        vectors = np.array([graph.get_node(nid)["vector"] for nid in node_ids])

        # Save graph structure as JSON alongside NPZ
        structure = {
            "nodes": [],
            "edges": [],
        }
        for nid in node_ids:
            data = graph.get_node(nid)
            structure["nodes"].append({
                "id": nid,
                "label": data["label"],
                "properties": data["properties"],
            })

        for src, tgt, key, edata in graph.edges():
            structure["edges"].append({
                "source": src,
                "target": tgt,
                "key": key,
                "relation": edata.get("relation", ""),
                "properties": edata.get("properties", {}),
            })

        npz_path = save_path.with_suffix(".npz")
        json_path = save_path.with_suffix(".json")

        np.savez(npz_path, vectors=vectors, node_ids=np.array(node_ids))

        with open(json_path, "w") as f:
            json.dump(structure, f, indent=2)

        return str(npz_path)

    elif format == "json":
        # Save everything as JSON (HD vectors as lists)
        data = {
            "nodes": [],
            "edges": [],
            "hd_dim": graph.hd.dim,
        }
        for nid in graph.nodes():
            ndata = graph.get_node(nid)
            data["nodes"].append({
                "id": nid,
                "label": ndata["label"],
                "properties": ndata["properties"],
                "vector": ndata["vector"].tolist(),
            })

        for src, tgt, key, edata in graph.edges():
            data["edges"].append({
                "source": src,
                "target": tgt,
                "key": key,
                "relation": edata.get("relation", ""),
                "properties": edata.get("properties", {}),
            })

        json_path = save_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        return str(json_path)

    else:
        raise ValueError(f"Unsupported format: {format}")


def load_graph(path: str, format: str = "npz") -> SpectralMemoryGraph:
    """Load a knowledge graph from disk.

    Args:
        path: Input file path.
        format: Serialization format ('npz' or 'json').

    Returns:
        Loaded SpectralMemoryGraph.
    """
    load_path = Path(path)

    if format == "npz":
        npz_path = load_path if load_path.suffix == ".npz" else load_path.with_suffix(".npz")
        json_path = load_path.with_suffix(".json")

        if not npz_path.exists():
            npz_path = load_path

        data = np.load(npz_path, allow_pickle=True)
        vectors = data["vectors"]
        node_ids = list(data["node_ids"])

        with open(json_path) as f:
            structure = json.load(f)

        graph = SpectralMemoryGraph(hd_dim=vectors.shape[1], seed=42)

        node_id_to_vector = {}
        for i, nid in enumerate(node_ids):
            node_id_to_vector[str(nid)] = vectors[i]

        for node_info in structure["nodes"]:
            nid = node_info["id"]
            graph.add_node(
                nid,
                label=node_info.get("label", ""),
                properties=node_info.get("properties", {}),
                vector=node_id_to_vector.get(nid),
            )

        for edge_info in structure["edges"]:
            graph.add_edge(
                edge_info["source"],
                edge_info["target"],
                edge_info.get("relation", ""),
                properties=edge_info.get("properties", {}),
            )

        return graph

    elif format == "json":
        json_path = load_path if load_path.suffix == ".json" else load_path.with_suffix(".json")

        with open(json_path) as f:
            data = json.load(f)

        hd_dim = data.get("hd_dim", 10000)
        graph = SpectralMemoryGraph(hd_dim=hd_dim, seed=42)

        for node_info in data["nodes"]:
            vector = np.array(node_info["vector"], dtype=np.int8)
            graph.add_node(
                node_info["id"],
                label=node_info.get("label", ""),
                properties=node_info.get("properties", {}),
                vector=vector,
            )

        for edge_info in data["edges"]:
            graph.add_edge(
                edge_info["source"],
                edge_info["target"],
                edge_info.get("relation", ""),
                properties=edge_info.get("properties", {}),
            )

        return graph

    else:
        raise ValueError(f"Unsupported format: {format}")


def save_checkpoint(
    graph: SpectralMemoryGraph,
    path: str,
) -> str:
    """Save a full SMGP checkpoint (graph + metadata).

    Args:
        graph: SpectralMemoryGraph to checkpoint.
        path: Checkpoint directory path.

    Returns:
        Path to the checkpoint directory.
    """
    from smgp.utils.io import save_graph
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_graph(graph, str(checkpoint_dir / "graph"), format="npz")
    return str(checkpoint_dir)


def load_checkpoint(path: str) -> SpectralMemoryGraph:
    """Load a SMGP checkpoint.

    Args:
        path: Checkpoint directory path.

    Returns:
        Loaded SpectralMemoryGraph.
    """
    from smgp.utils.io import load_graph
    return load_graph(str(Path(path) / "graph"), format="npz")
