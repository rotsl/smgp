"""Core mathematical modules for the Spectral Memory Graph Processor.

This subpackage implements the foundational mathematical primitives:
- Directed, multi-relational, typed property graphs with HD addressing
- Spectral graph transforms, wavelets, and Laplacian decompositions
- Topological persistence analysis and graph stability
- Category-theoretic graph rewriting (DPO)
- Hyperdimensional vector algebra for O(1) associative recall
"""
from smgp.core.category import GraphRewriter
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.hyperdim import HyperdimensionalMemory
from smgp.core.spectral import SpectralMethods
from smgp.core.topology import TopologicalAnalyzer

__all__ = [
    "SpectralMemoryGraph",
    "SpectralMethods",
    "TopologicalAnalyzer",
    "GraphRewriter",
    "HyperdimensionalMemory",
]
