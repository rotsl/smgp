"""SMGP Hardware Abstraction Layer.

Provides transparent hardware acceleration for SMGP operations.
Use by setting backend='hardware' in SMGPConfig.

References:
    Hennessy, J.L. & Patterson, D.A. (2019). "Computer Architecture." 6th Ed.
"""

from smgp_hal.hw_session import HWSession
from smgp_hal.executor import HWExecutor
from smgp_hal.memory_mapper import MemoryMapper

__all__ = ["HWSession", "HWExecutor", "MemoryMapper"]
__version__ = "0.1.0"
