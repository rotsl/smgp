"""Multiscale spectral attention mechanism for SMGP.

Replaces standard transformer attention with graph spectral operations,
achieving O(N log N) complexity on structured inputs.
"""
from smgp.attention.spectral_attn import SpectralAttention

__all__ = ["SpectralAttention"]
