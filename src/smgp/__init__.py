"""Spectral Memory Graph Processor (SMGP) — persistent, hallucination-free AI reasoning.

Reference: R, R. (2026). Spectral Memory Graph Processing for Persistent AI Memory.
"""

from smgp.config import SMGPConfig
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.hyperdim import HyperdimensionalMemory

__version__ = "1.0.0"
__all__ = [
    "SMGPConfig",
    "SpectralMemoryGraph",
    "HyperdimensionalMemory",
]
