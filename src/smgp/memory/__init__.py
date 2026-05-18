"""Persistent memory subsystem for SMGP.

Provides non-volatile memory abstraction, hyperdimensional associative recall,
and lifecycle management with topological pruning for controlled forgetting.
"""
from smgp.memory.associative import AssociativeMemory
from smgp.memory.lifecycle import MemoryLifecycle
from smgp.memory.store import MemoryStore

__all__ = ["MemoryStore", "AssociativeMemory", "MemoryLifecycle"]
