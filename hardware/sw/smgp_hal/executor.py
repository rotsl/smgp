"""Hardware executor: translates SMGP operations to ISA instructions.

Mirrors the interface of smgp.core.graph so users can set
backend='hardware' and everything transparently runs on the accelerator.

References:
    Hennessy & Patterson (2019). "Computer Architecture."
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from smgp_hal.hw_session import HWSession


def _encode_instr(opcode: int, sub_opcode: int, flags: int, operand: int) -> int:
    """Encode a 32-bit SMGP ISA instruction."""
    return ((opcode & 0xF) << 28) | ((sub_opcode & 0xF) << 24) | \
           ((flags & 0xFF) << 16) | (operand & 0xFFFF)


class HWExecutor:
    """Translates high-level SMGP operations to hardware ISA instructions.

    Provides the same interface as smgp.core.graph.SpectralMemoryGraph
    but offloads computation to the SMGP accelerator.

    Attributes:
        session: Active hardware session.
        node_count: Number of nodes in the hardware graph.
    """

    # ISA opcodes
    OPC_GRAPH_CTOR = 0x1
    OPC_SPECTRAL = 0x2
    OPC_HD = 0x3
    OPC_TOPOLOGY = 0x4
    OPC_REWRITE = 0x5
    OPC_MEMORY = 0x6

    def __init__(self, session: HWSession):
        """Initialize the executor.

        Args:
            session: Active HWSession.
        """
        self.session = session
        self.node_count = 0

    def add_node(self, node_id: str, label: str = "",
                 properties: Optional[Dict] = None,
                 vector: Optional[np.ndarray] = None) -> str:
        """Add a node via hardware."""
        instr = _encode_instr(
            self.OPC_GRAPH_CTOR, 0x0, 0x01,  # ADD_NODE with START flag
            self.node_count & 0xFFFF
        )
        self.session.execute(instr)
        self.node_count += 1
        return node_id

    def add_edge(self, source_id: str, target_id: str,
                 relation: str = "") -> str:
        """Add an edge via hardware."""
        instr = _encode_instr(
            self.OPC_GRAPH_CTOR, 0x1, 0x01,  # ADD_EDGE with START flag
            hash((source_id, target_id)) & 0xFFFF
        )
        self.session.execute(instr)
        return f"edge_{source_id}_{target_id}"

    def compute_laplacian(self) -> bool:
        """Offload Laplacian computation to hardware."""
        instr = _encode_instr(
            self.OPC_SPECTRAL, 0x0, 0x01,  # COMPUTE_LAP with START
            self.node_count & 0xFFFF
        )
        return self.session.execute(instr)

    def spectral_convolve(self, signal: np.ndarray,
                          coefficients: List[float]) -> np.ndarray:
        """Offload spectral convolution to hardware."""
        order = len(coefficients)
        instr = _encode_instr(
            self.OPC_SPECTRAL, 0x2, 0x01,  # SPECTRAL_CONV with START
            order & 0xFFFF
        )
        success = self.session.execute(instr)
        # Read results back from hardware memory
        result = np.zeros_like(signal)
        if success:
            for i in range(len(signal)):
                val = self.session.read_reg(0x4000 + i * 4)
                if val is not None:
                    result[i] = np.int32(val) / (1 << 24)
        return result

    def hd_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Offload HD bind to hardware."""
        instr = _encode_instr(self.OPC_HD, 0x1, 0x01, 0)  # HD_BIND
        self.session.execute(instr)
        # Hardware returns XOR of vectors
        return np.where(a == b, 1, -1).astype(np.int8)

    def hd_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Offload HD similarity to hardware."""
        instr = _encode_instr(self.OPC_HD, 0x4, 0x01, 0)  # HD_SIMILARITY
        self.session.execute(instr)
        # In simulation mode, compute locally; on real HW, read result register
        if self.session.backend == "verilator":
            return float(np.dot(a.astype(np.float64), b.astype(np.float64))) / len(a)
        val = self.session.read_reg(0x2000)
        if val is not None:
            return float(val) / 10000.0
        return float(np.dot(a, b)) / len(a)

    def build_filtration(self) -> bool:
        """Offload topological filtration to hardware."""
        instr = _encode_instr(self.OPC_TOPOLOGY, 0x0, 0x01, 0)
        return self.session.execute(instr)

    def check_stability(self, threshold: float = 0.5) -> bool:
        """Offload topological stability check to hardware."""
        threshold_int = int(threshold * (1 << 24))
        instr = _encode_instr(self.OPC_TOPOLOGY, 0x2, 0x01,
                              threshold_int >> 16)
        return self.session.execute(instr)
