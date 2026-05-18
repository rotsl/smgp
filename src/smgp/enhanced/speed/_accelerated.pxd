# Cython declaration file for _accelerated.pyx
"""Type declarations for the Cython-accelerated SMGP routines.

This file allows other Cython modules to ``cimport`` the functions
defined in ``_accelerated.pyx`` with full type information.
"""
cimport numpy as cnp


def fast_eigen_decompose(cnp.ndarray adj_matrix, int k):
    """Return (eigenvalues, eigenvectors) for the *k* smallest eigenvalues."""
    pass


def fast_hd_bind(cnp.ndarray a, cnp.ndarray b):
    """Element-wise multiply of two int8 HD vectors."""
    pass


def fast_hd_similarity(cnp.ndarray a, cnp.ndarray b):
    """Cosine similarity between two vectors (float64)."""
    pass


def fast_graph_traversal(cnp.ndarray adj_matrix, int start, int max_depth):
    """BFS traversal returning ordered list of visited node indices."""
    pass
