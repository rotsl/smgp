"""Optional Cython-accelerated routines with pure-Python fallback."""

from smgp.enhanced.speed.accelerated import (
    fast_eigen_decompose,
    fast_graph_traversal,
    fast_hd_bind,
    fast_hd_similarity,
)

__all__ = [
    "fast_eigen_decompose",
    "fast_graph_traversal",
    "fast_hd_bind",
    "fast_hd_similarity",
]
