# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated routines for SMGP core operations.

Provides fast implementations of:
  - Eigendecomposition of sparse adjacency matrices
  - Hyperdimensional bind/unbind operations
  - Cosine similarity computation
  - Breadth-first graph traversal

Compile with Cython >= 0.29 and NumPy >= 1.20.  OpenMP is used
where available for parallel loops.
"""
from __future__ import annotations

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()

# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def fast_eigen_decompose(cnp.ndarray adj_matrix, int k):
    """Compute the *k* smallest eigenvalues and eigenvectors.

    Parameters
    ----------
    adj_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix (converted to dense internally for the
        LAPACK call inside ``numpy.linalg.eigh``).
    k : int
        Number of eigenvalues to return.

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
    eigenvectors : np.ndarray, shape (n, k)
    """
    cdef int n = adj_matrix.shape[0]

    # Convert sparse → dense for the dense LAPACK eigen-solver.
    if hasattr(adj_matrix, "toarray"):
        dense = np.asarray(adj_matrix.toarray(), dtype=np.float64)
    else:
        dense = np.asarray(adj_matrix, dtype=np.float64)

    # Compute full spectrum then truncate
    cdef cnp.ndarray eigvals_full
    cdef cnp.ndarray eigvecs_full
    eigvals_full, eigvecs_full = np.linalg.eigh(dense)

    cdef int actual_k = min(k, n)
    return eigvals_full[:actual_k].copy(), eigvecs_full[:, :actual_k].copy()


# ---------------------------------------------------------------------------
# Hyperdimensional bind (element-wise multiply of bipolar / int8 vectors)
# ---------------------------------------------------------------------------

def fast_hd_bind(cnp.ndarray a, cnp.ndarray b):
    """Element-wise multiply of two int8 HD vectors.

    Parameters
    ----------
    a, b : np.ndarray, shape (D,), dtype int8

    Returns
    -------
    np.ndarray, shape (D,), dtype int8
    """
    # Work in int16 to avoid overflow before casting back to int8
    cdef cnp.ndarray[cnp.int16_t, ndim=1] result_i16 = (
        a.astype(np.int16) * b.astype(np.int16)
    )
    # Clamp to [-1, 1] to stay bipolar
    cdef cnp.ndarray result = np.sign(result_i16).astype(np.int8)
    return result


# ---------------------------------------------------------------------------
# Hyperdimensional cosine similarity
# ---------------------------------------------------------------------------

def fast_hd_similarity(cnp.ndarray a, cnp.ndarray b):
    """Cosine similarity between two vectors using float64 dot product.

    Parameters
    ----------
    a, b : array-like

    Returns
    -------
    float
        Cosine similarity in [-1, 1].
    """
    cdef cnp.ndarray[double, ndim=1] fa = np.ascontiguousarray(a, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] fb = np.ascontiguousarray(b, dtype=np.float64)

    cdef double dot = np.dot(fa, fb)
    cdef double norm_a = sqrt(np.dot(fa, fa))
    cdef double norm_b = sqrt(np.dot(fb, fb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    cdef double sim = dot / (norm_a * norm_b)
    # Clamp to [-1, 1] for numerical safety
    if sim > 1.0:
        sim = 1.0
    elif sim < -1.0:
        sim = -1.0
    return sim


# ---------------------------------------------------------------------------
# BFS graph traversal
# ---------------------------------------------------------------------------

def fast_graph_traversal(cnp.ndarray adj_matrix, int start, int max_depth):
    """Breadth-first traversal over a dense/sparse adjacency matrix.

    Parameters
    ----------
    adj_matrix : array-like, shape (n, n)
        Adjacency matrix (any numeric type, converted to dense).
    start : int
        Index of the starting node.
    max_depth : int
        Maximum BFS depth.

    Returns
    -------
    visited : list[int]
        Ordered list of visited node indices.
    """
    if hasattr(adj_matrix, "toarray"):
        dense = np.asarray(adj_matrix.toarray(), dtype=np.int32)
    else:
        dense = np.asarray(adj_matrix, dtype=np.int32)

    cdef int n = dense.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] seen = np.zeros(n, dtype=np.uint8)
    seen[start] = 1

    cdef list queue = [start]
    cdef list visited = [start]
    cdef int depth = 0
    cdef int qi = 0
    cdef int level_size = 1
    cdef int new_level_size = 0
    cdef int node, neighbor

    while qi < len(queue) and depth < max_depth:
        level_size = len(queue) - qi
        new_level_size = 0
        for _ in range(level_size):
            node = queue[qi]
            qi += 1
            for neighbor in range(n):
                if dense[node, neighbor] != 0 and seen[neighbor] == 0:
                    seen[neighbor] = 1
                    queue.append(neighbor)
                    visited.append(neighbor)
                    new_level_size += 1
        depth += 1
        if new_level_size == 0:
            break

    return visited
