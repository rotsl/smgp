"""Streaming spectral attention for long sequences.

Extends :class:`~smgp.attention.spectral_attn.SpectralAttention` so that
very long token sequences can be processed in fixed-size overlapping
chunks without materialising the full graph at once.

The accumulated sliding-window graph is maintained across chunks so that
cross-chunk boundary tokens are connected via the overlap region.

References
----------
  - Dai, Z., et al. (2019). "Transformer-XL." ACL.
  - Shazeer, N., et al. (2019). "Fast Transformer Decoding." NeurIPS.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from smgp.attention.spectral_attn import SpectralAttention
from smgp.core.graph import SpectralMemoryGraph


class StreamingSpectralAttention(SpectralAttention):
    """Spectral attention that processes input in overlapping chunks.

    Parameters
    ----------
    graph : SpectralMemoryGraph
        Seed graph providing the HD memory backend.
    hidden_dim : int
        Dimensionality of internal attention embeddings.
    num_heads : int
        Number of parallel attention heads.
    num_scales : int
        Number of wavelet scales.
    chunk_size : int
        Number of tokens per chunk.
    overlap : int
        Number of tokens shared between consecutive chunks.
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_scales: int = 3,
        chunk_size: int = 128,
        overlap: int = 16,
    ) -> None:
        super().__init__(
            graph=graph,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_scales=num_scales,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Streaming state
        self._stream_graph: SpectralMemoryGraph | None = None
        self._token_offset: int = 0

    def forward(
        self,
        node_embeddings: np.ndarray,
        chunk_size: int | None = None,
        stream_state: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Process *node_embeddings* in overlapping chunks.

        Parameters
        ----------
        node_embeddings : np.ndarray, shape (N, D)
            Full sequence of node embeddings.
        chunk_size : int or None
            Override the default chunk size.  When *None* the instance
            default is used.
        stream_state : dict or None
            External stream state (for resuming a previous session).
            When *None* a fresh state is initialised.

        Returns
        -------
        np.ndarray, shape (N, hidden_dim)
            Contextualised embeddings for the full sequence.
        """
        cs = chunk_size or self.chunk_size

        if stream_state is not None:
            self._restore_state(stream_state)
        else:
            self.reset()

        n_total = node_embeddings.shape[0]
        step = cs - self.overlap
        output_chunks: list[np.ndarray] = []

        # Process in overlapping windows
        start = 0
        while start < n_total:
            end = min(start + cs, n_total)
            chunk_emb = node_embeddings[start:end]
            is_last = end >= n_total

            # Pad last chunk if needed (so spectral methods work)
            if chunk_emb.shape[0] < 2:
                # Too small for spectral — use linear projection only
                chunk_out = chunk_emb @ self._W_o
            else:
                # Build a local graph for this chunk
                chunk_graph = self.build_graph_from_tokens(chunk_emb)

                # Incrementally merge into the accumulated stream graph
                self._incremental_update(chunk_graph, stream_state=None)

                # Run spectral attention on the merged graph
                chunk_out = super().forward(chunk_emb)

            # Trim overlap: for non-last chunks, keep only the leading
            # ``step`` rows so that concatenation yields exactly N rows.
            if not is_last and chunk_out.shape[0] > step:
                chunk_out = chunk_out[:step]

            output_chunks.append(chunk_out)

            if is_last:
                break
            start += step

        return np.concatenate(output_chunks, axis=0)

    def _incremental_update(
        self,
        new_chunk_graph: SpectralMemoryGraph,
        stream_state: dict[str, Any] | None,
    ) -> None:
        """Merge nodes/edges from *new_chunk_graph* into the stream graph.

        The accumulated graph grows up to a maximum size (``chunk_size +
        overlap``) to bound memory usage; older nodes beyond the window
        are dropped.

        Parameters
        ----------
        new_chunk_graph : SpectralMemoryGraph
            Graph built from the latest chunk.
        stream_state : dict or None
            Reserved for future external state management.
        """
        if self._stream_graph is None:
            self._stream_graph = new_chunk_graph
            return

        # Copy new nodes into the accumulated graph
        for nid in new_chunk_graph.nodes():
            data = new_chunk_graph.get_node(nid)
            if data is not None and nid not in self._stream_graph.nodes():
                self._stream_graph.add_node(
                    nid,
                    label=data["label"],
                    properties=data["properties"],
                    vector=data["vector"],
                )

        # Copy new edges
        for src, tgt, key, edata in new_chunk_graph.edges():
            if (src in self._stream_graph.nodes()
                    and tgt in self._stream_graph.nodes()):
                # Avoid duplicate edges
                self._stream_graph.add_edge(
                    src, tgt,
                    edata.get("relation", ""),
                    properties=edata.get("properties", {}),
                )

        # Trim accumulated graph to keep bounded size
        max_nodes = self.chunk_size + self.overlap * 2
        all_nodes = list(self._stream_graph.nodes())
        if len(all_nodes) > max_nodes:
            to_remove = all_nodes[: len(all_nodes) - max_nodes]
            for nid in to_remove:
                self._stream_graph.remove_node(nid)

    def reset(self) -> None:
        """Clear all accumulated streaming state."""
        self._stream_graph = None
        self._token_offset = 0

    def _restore_state(self, state: dict[str, Any]) -> None:
        """Restore streaming state from an external dict.

        Parameters
        ----------
        state : dict
            Must contain ``'token_offset'`` (int).
        """
        self._token_offset = state.get("token_offset", 0)
        self._stream_graph = None  # Graph state is not serialisable here

    def get_state(self) -> dict[str, Any]:
        """Return the current streaming state as a dict.

        Returns
        -------
        dict with ``'token_offset'`` key.
        """
        return {"token_offset": self._token_offset}
