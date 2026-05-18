"""
PYNQ-based Hardware Abstraction Layer for SMGPU

Provides an alternative HAL backend that uses the PYNQ framework to control
the SMGPU accelerator on Zynq / Zynq UltraScale+ boards.

This module lazy-imports ``pynq`` — if the package is not available, an
ImportError with a helpful message is raised at runtime.
"""

from typing import Dict, List, Optional


class PYNQHAL:
    """Alternative HAL backend using PYNQ for Zynq boards.

    Implements the same conceptual interface as ``hardware.sw.smgp_hal.executor``
    but targets the PYNQ runtime instead of PCIe / simulation backends.

    Usage::

        try:
            hal = PYNQHAL("smgpu.bit")
            result = hal.execute(opcode=0x01, sub_opcode=0, flags=0, operand=1024)
            data = hal.read_buffer(address=0x1000, length=256)
            hal.close()
        except ImportError as e:
            print(f"PYNQ not available: {e}")
    """

    # Opcodes (matching smgp_isa_pkg.sv)
    OP_NOP                  = 0x00
    OP_COMPUTE_LAPLACIAN    = 0x01
    OP_EIGEN_DECOMPOSITION  = 0x02
    OP_HD_BIND              = 0x03
    OP_GRAPH_REWRITE        = 0x04
    OP_DMA_READ             = 0x80
    OP_DMA_WRITE            = 0x81

    def __init__(self, overlay_path: str = "smgpu.bit"):
        """Initialise the PYNQ HAL.

        Parameters
        ----------
        overlay_path:
            Path to the SMGPU bitstream file.

        Raises
        ------
        ImportError
            If PYNQ is not installed.
        """
        self._overlay_path = overlay_path
        self._overlay = None
        self._dma = None
        self._registers = None
        self._is_open = False

        # Lazy import
        try:
            from pynq import Overlay  # noqa: F401
            self._Overlay = Overlay
        except ImportError:
            raise ImportError(
                "PYNQ runtime is not installed on this platform. "
                "PYNQHAL requires a Xilinx Zynq board with the PYNQ "
                "framework. Install with: pip install pynq"
            )

        self._load_overlay(overlay_path)
        self._is_open = True

    def _load_overlay(self, overlay_path: str) -> None:
        """Load the PYNQ overlay and discover IP cores."""
        self._overlay = self._Overlay(overlay_path)

        # Discover DMA engine
        for attr_name in ("axi_dma_0", "dma", "axi_dma", "smgp_dma"):
            if hasattr(self._overlay, attr_name):
                self._dma = getattr(self._overlay, attr_name)
                break

        # Discover register interface (for opcode dispatch)
        for attr_name in ("smgp_core", "smgp_core_0", "core", "accelerator"):
            if hasattr(self._overlay, attr_name):
                self._registers = getattr(self._overlay, attr_name)
                break

        print(f"[PYNQHAL] Overlay loaded: {overlay_path}")
        if self._dma is None:
            print("[PYNQHAL] WARNING: No DMA engine found in overlay.")
        if self._registers is None:
            print("[PYNQHAL] WARNING: No register interface found in overlay.")

    def execute(
        self,
        opcode: int,
        sub_opcode: int = 0,
        flags: int = 0,
        operand: int = 0,
    ) -> Dict:
        """Dispatch an instruction to the SMGPU accelerator.

        Parameters
        ----------
        opcode:
            SMGP operation code (e.g. ``OP_COMPUTE_LAPLACIAN``).
        sub_opcode:
            Sub-operation selector.
        flags:
            Control flags (reserved, set to 0).
        operand:
            Operand data (e.g. nnz, k, address).

        Returns
        -------
        dict
            Execution result with keys:
            ``opcode``, ``sub_opcode``, ``flags``, ``operand``,
            ``status``, ``cycles``, ``error``.
        """
        if not self._is_open:
            raise RuntimeError("HAL is closed. Create a new PYNQHAL instance.")

        # Pack the instruction word (32-bit, matching the ISA encoding)
        instr_word = (
            ((opcode & 0xFF) << 24) |
            ((sub_opcode & 0xFF) << 16) |
            ((flags & 0xFFFF) << 0)
        )

        status = "OK"
        cycles = 0
        error = None

        try:
            if self._registers is not None:
                # Write opcode register
                self._registers.write(0x00, opcode)
                # Write sub-opcode register
                self._registers.write(0x04, sub_opcode)
                # Write operand register
                self._registers.write(0x08, operand)
                # Write flags and start
                self._registers.write(0x0C, flags | 0x01)

                # Poll status register (with timeout)
                import time
                timeout_s = 10.0
                elapsed = 0.0
                poll_interval = 0.001  # 1 ms

                while elapsed < timeout_s:
                    done = self._registers.read(0x10) & 0x01
                    if done:
                        cycles = self._registers.read(0x14)
                        error_code = (self._registers.read(0x10) >> 1) & 0x7F
                        if error_code != 0:
                            error = f"Accelerator error code: {error_code}"
                            status = "ERROR"
                        break
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                else:
                    status = "TIMEOUT"
                    error = f"Execution timed out after {timeout_s}s"
            else:
                # No register interface — software emulation mode
                cycles = self._emulate_cycles(opcode, operand)
                status = "EMULATED"
        except Exception as e:
            status = "ERROR"
            error = str(e)

        return {
            "opcode": opcode,
            "sub_opcode": sub_opcode,
            "flags": flags,
            "operand": operand,
            "status": status,
            "cycles": cycles,
            "error": error,
        }

    def read_buffer(self, address: int, length: int) -> List:
        """Read data from the accelerator's memory-mapped buffer.

        Parameters
        ----------
        address:
            Physical or virtual address offset.
        length:
            Number of 32-bit words to read.

        Returns
        -------
        list of int
            The read data words.
        """
        if self._registers is None:
            raise RuntimeError("No register interface available for memory reads.")

        data = []
        for i in range(length):
            word = self._registers.read(address + i * 4)
            data.append(word)
        return data

    def write_buffer(self, address: int, data: List) -> None:
        """Write data to the accelerator's memory-mapped buffer.

        Parameters
        ----------
        address:
            Physical or virtual address offset.
        data:
            List of 32-bit words to write.
        """
        if self._registers is None:
            raise RuntimeError("No register interface available for memory writes.")

        for i, word in enumerate(data):
            self._registers.write(address + i * 4, word)

    def dma_transfer(
        self,
        src_buffer,
        dst_buffer=None,
        length: Optional[int] = None,
    ) -> None:
        """Perform a DMA transfer between host memory and the accelerator.

        Parameters
        ----------
        src_buffer:
            Source PYNQ buffer.
        dst_buffer:
            Destination PYNQ buffer (if *None*, transfer is device->host read).
        length:
            Number of elements to transfer.
        """
        if self._dma is None:
            raise RuntimeError("No DMA engine available.")

        if dst_buffer is None:
            # Device to host
            self._dma.recvchannel.transfer(src_buffer, length=length)
            self._dma.recvchannel.wait()
        else:
            # Host to device
            self._dma.sendchannel.transfer(src_buffer, length=length)
            self._dma.sendchannel.wait()

    def close(self) -> None:
        """Release all resources and close the HAL."""
        if self._overlay is not None:
            # PYNQ Overlay doesn't have an explicit close in all versions
            self._overlay = None
        self._dma = None
        self._registers = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Return *True* if the HAL is active."""
        return self._is_open

    def __enter__(self) -> "PYNQHAL":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _emulate_cycles(opcode: int, operand: int) -> int:
        """Software-only cycle estimation (mirrors cycle_simulator.py)."""
        if opcode == PYNQHAL.OP_COMPUTE_LAPLACIAN:
            return operand  # nnz cycles
        elif opcode == PYNQHAL.OP_EIGEN_DECOMPOSITION:
            return operand  # k * nnz
        elif opcode == PYNQHAL.OP_HD_BIND:
            return 1
        elif opcode == PYNQHAL.OP_GRAPH_REWRITE:
            return 10 + (operand // 10) if operand > 0 else 10
        elif opcode in (PYNQHAL.OP_DMA_READ, PYNQHAL.OP_DMA_WRITE):
            return operand + 2
        else:
            return 1
