"""Public wrapper around the Cython-accelerated SMGP routines.

On import this module attempts to load the compiled Cython extension.
If that fails (e.g. Cython was not installed at build time) it falls
back to pure-Python / NumPy implementations and emits a warning.
"""
from __future__ import annotations

import warnings

# Try the compiled extension first; fall back to pure Python.
try:
    from smgp.enhanced.speed._accelerated import (  # type: ignore[import-untyped]
        fast_eigen_decompose,
        fast_graph_traversal,
        fast_hd_bind,
        fast_hd_similarity,
    )
except ImportError:
    warnings.warn(
        "Cython extension not available, using pure Python fallback",
        stacklevel=2,
    )

    # ------------------------------------------------------------------
    # Pure-Python fallback implementations
    # ------------------------------------------------------------------

    def fast_eigen_decompose(adj_matrix, k):  # type: ignore[misc]
        """Pure-Python fallback for eigendecomposition.

        Converts the adjacency matrix to a dense NumPy array and calls
        ``numpy.linalg.eigh``.  Returns the *k* smallest eigenvalues and
        their corresponding eigenvectors.

        Parameters
        ----------
        adj_matrix : scipy.sparse matrix or array-like
        k : int

        Returns
        -------
        eigenvalues : np.ndarray, shape (k,)
        eigenvectors : np.ndarray, shape (n, k)
        """
        import numpy as np

        if hasattr(adj_matrix, "toarray"):
            dense = np.asarray(adj_matrix.toarray(), dtype=np.float64)
        else:
            dense = np.asarray(adj_matrix, dtype=np.float64)

        eigvals, eigvecs = np.linalg.eigh(dense)
        actual_k = min(k, dense.shape[0])
        return eigvals[:actual_k].copy(), eigvecs[:, :actual_k].copy()

    def fast_hd_bind(a, b):  # type: ignore[misc]
        """Pure-Python fallback for HD bind (element-wise multiply).

        Parameters
        ----------
        a, b : array-like, int8

        Returns
        -------
        np.ndarray, dtype int8 — element-wise product, sign-quantised.
        """
        import numpy as np

        a16 = np.asarray(a, dtype=np.int16)
        b16 = np.asarray(b, dtype=np.int16)
        return np.sign(a16 * b16).astype(np.int8)

    def fast_hd_similarity(a, b):  # type: ignore[misc]
        """Pure-Python fallback for cosine similarity.

        Parameters
        ----------
        a, b : array-like

        Returns
        -------
        float — cosine similarity in [-1, 1].
        """
        import numpy as np

        fa = np.asarray(a, dtype=np.float64).ravel()
        fb = np.asarray(b, dtype=np.float64).ravel()
        dot = np.dot(fa, fb)
        norm_a = np.linalg.norm(fa)
        norm_b = np.linalg.norm(fb)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))

    def fast_graph_traversal(adj_matrix, start, max_depth):  # type: ignore[misc]
        """Pure-Python fallback for BFS graph traversal.

        Parameters
        ----------
        adj_matrix : scipy.sparse matrix or array-like, shape (n, n)
        start : int
        max_depth : int

        Returns
        -------
        list[int] — ordered visited node indices.
        """
        import numpy as np

        if hasattr(adj_matrix, "toarray"):
            dense = np.asarray(adj_matrix.toarray(), dtype=np.int32)
        else:
            dense = np.asarray(adj_matrix, dtype=np.int32)

        n = dense.shape[0]
        visited = []
        seen = set()
        queue = [(start, 0)]
        seen.add(start)

        while queue:
            node, depth = queue.pop(0)
            if depth > max_depth:
                continue
            visited.append(node)
            for neighbor in range(n):
                if dense[node, neighbor] != 0 and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return visited


__all__ = [
    "fast_eigen_decompose",
    "fast_hd_bind",
    "fast_hd_similarity",
    "fast_graph_traversal",
]
