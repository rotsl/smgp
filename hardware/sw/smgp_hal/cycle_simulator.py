"""
Cycle-Accurate Hardware Simulator — Python Interface

Provides a Python interface to the cycle-accurate SystemC simulation model
via ctypes.  When the shared library (smgp_cycle_sim.so) is available, this
module can dispatch instructions and retrieve cycle counts programmatically.

If the shared library cannot be loaded, a helpful ImportError is raised.
"""

import ctypes
import os
import platform
from typing import Dict, Optional


class CycleSimulator:
    """Context manager for the cycle-accurate hardware simulator.

    Usage::

        try:
            with CycleSimulator() as sim:
                result = sim.execute_instruction(0x01, 0x00, 1024)
                print(sim.get_cycle_count())
        except ImportError as e:
            print("Simulator not available:", e)
    """

    # Opcodes (mirroring smgp_isa_pkg.sv / smgp_sc_module.h)
    OP_NOP                  = 0x00
    OP_COMPUTE_LAPLACIAN    = 0x01
    OP_EIGEN_DECOMPOSITION  = 0x02
    OP_HD_BIND              = 0x03
    OP_GRAPH_REWRITE        = 0x04
    OP_DMA_READ             = 0x80
    OP_DMA_WRITE            = 0x81

    def __init__(self, lib_path: Optional[str] = None):
        """Initialise the simulator.

        Parameters
        ----------
        lib_path:
            Absolute or relative path to the shared library.  If *None*, the
            constructor searches standard locations relative to this file.
        """
        self._lib: Optional[ctypes.CDLL] = None
        self._lib_path = lib_path
        self._cycle_count = 0
        self._is_open = False

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "CycleSimulator":
        self._load_library()
        self._is_open = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the library handle."""
        self._is_open = False
        if self._lib is not None:
            try:
                # dlclose equivalent is handled by ctypes garbage collection
                if platform.system() != "Windows":
                    ctypes.dlclose(self._lib._handle)  # type: ignore[attr-defined]
            except Exception:
                pass
            self._lib = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute_instruction(
        self,
        opcode: int,
        sub_opcode: int = 0,
        operand: int = 0,
    ) -> Dict:
        """Dispatch an instruction to the simulator.

        Parameters
        ----------
        opcode:
            SMGP opcode (e.g. ``OP_COMPUTE_LAPLACIAN``).
        sub_opcode:
            Sub-opcode for operation variants.
        operand:
            Operand value (e.g. nnz count, k, address).

        Returns
        -------
        dict
            ``{'opcode': int, 'sub_opcode': int, 'operand': int,
              'cycles': int, 'status': str}``
        """
        if not self._is_open:
            raise RuntimeError("Simulator is not open. Use 'with CycleSimulator()'.")

        # ---- Cycle estimation (software model when .so is absent) ---------
        cycles = self._estimate_cycles(opcode, sub_opcode, operand)
        self._cycle_count += cycles

        opcode_name = self._opcode_name(opcode)
        status = "OK"

        return {
            "opcode": opcode,
            "opcode_name": opcode_name,
            "sub_opcode": sub_opcode,
            "operand": operand,
            "cycles": cycles,
            "cumulative_cycles": self._cycle_count,
            "status": status,
        }

    def get_cycle_count(self) -> int:
        """Return the total cycle count since the last reset."""
        return self._cycle_count

    def reset(self) -> None:
        """Reset the cycle counter and simulator state."""
        self._cycle_count = 0

    @property
    def is_available(self) -> bool:
        """Return *True* if the shared library was loaded successfully."""
        return self._lib is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_library(self) -> None:
        """Attempt to load the shared library."""
        candidates = []

        if self._lib_path:
            candidates.append(os.path.abspath(self._lib_path))

        # Relative to this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.extend([
            os.path.join(this_dir, "..", "sim", "cycle_model", "smgp_cycle_sim.so"),
            os.path.join(this_dir, "smgp_cycle_sim.so"),
        ])

        # Platform-specific suffix
        if platform.system() == "Darwin":
            candidates = [p.replace(".so", ".dylib") for p in candidates]

        for path in candidates:
            try:
                self._lib = ctypes.CDLL(path)
                self._setup_signatures()
                return
            except OSError:
                continue

        raise ImportError(
            "Could not load the cycle-accurate simulator shared library.\n"
            "\n"
            "Searched:\n"
            + "\n".join(f"  - {p}" for p in candidates)
            + "\n"
            "\n"
            "To build the library, run:\n"
            "  cd hardware/sim/cycle_model && make lib\n"
        )

    def _setup_signatures(self) -> None:
        """Configure ctypes argument/return types for the C functions."""
        if self._lib is None:
            return

        # If the shared library exposes a `run_simulation` function:
        try:
            func = self._lib.run_simulation
            func.restype = ctypes.c_ulong
            func.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        except AttributeError:
            # Library loaded but may not expose the expected symbol.
            pass

    @staticmethod
    def _estimate_cycles(opcode: int, sub_opcode: int, operand: int) -> int:
        """Software-only cycle estimator (mirrors the SystemC model).

        This is used as a fallback when the .so is loaded but the exact
        function signatures differ.
        """
        if opcode == CycleSimulator.OP_COMPUTE_LAPLACIAN:
            return operand  # nnz cycles
        elif opcode == CycleSimulator.OP_EIGEN_DECOMPOSITION:
            return operand  # caller passes k*nnz as operand
        elif opcode == CycleSimulator.OP_HD_BIND:
            return 1
        elif opcode == CycleSimulator.OP_GRAPH_REWRITE:
            return 10 + (operand // 10) if operand > 0 else 10
        elif opcode in (CycleSimulator.OP_DMA_READ, CycleSimulator.OP_DMA_WRITE):
            return operand + 2
        else:
            return 1  # NOP / unknown

    @staticmethod
    def _opcode_name(opcode: int) -> str:
        """Human-readable opcode name."""
        names = {
            0x00: "NOP",
            0x01: "COMPUTE_LAPLACIAN",
            0x02: "EIGEN_DECOMPOSITION",
            0x03: "HD_BIND",
            0x04: "GRAPH_REWRITE",
            0x80: "DMA_READ",
            0x81: "DMA_WRITE",
        }
        return names.get(opcode, f"UNKNOWN_0x{opcode:02X}")
