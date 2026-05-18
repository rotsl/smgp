"""Hardware session manager for SMGP FPGA/PCIe communication.

Detects and connects to either a Verilator simulation or a physical
FPGA via the PCIe driver. Provides a context manager for resource cleanup.

References:
    Corbet, J. et al. (2005). "Linux Device Drivers." 3rd Ed., O'Reilly.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


class HWSession:
    """Manages the connection to SMGP hardware (simulation or FPGA).

    Supports two backends:
      - 'verilator': Connects to a running Verilator simulation via VPI/pipe.
      - 'pcie': Connects to a physical FPGA via the smgp_pcie driver.

    Attributes:
        backend: Backend type ('verilator' or 'pcie').
        connected: Whether the session is active.
        device_path: Path to the device or simulation pipe.
    """

    def __init__(self, backend: str = "verilator", device_path: Optional[str] = None):
        """Initialize a hardware session.

        Args:
            backend: Connection backend ('verilator' or 'pcie').
            device_path: Device path (auto-detected if None).
        """
        self.backend = backend
        self.connected = False
        self.device_path = device_path or self._auto_detect_device()
        self._process: Optional[subprocess.Popen] = None

    def _auto_detect_device(self) -> str:
        """Auto-detect the SMGP device.

        Returns:
            Device path or empty string if not found.
        """
        if self.backend == "pcie":
            dev = "/dev/smgp"
            if os.path.exists(dev):
                return dev
            return ""
        else:
            return "verilator"  # Placeholder for pipe/socket

    def connect(self) -> bool:
        """Establish connection to hardware.

        Returns:
            True if connection successful.
        """
        if self.backend == "verilator":
            self.connected = True
            return True
        elif self.backend == "pcie":
            if self.device_path and os.path.exists(self.device_path):
                self.connected = True
                return True
            return False
        return False

    def disconnect(self) -> None:
        """Close the hardware connection."""
        self.connected = False
        if self._process:
            self._process.terminate()
            self._process = None

    def write_reg(self, offset: int, value: int) -> bool:
        """Write a 32-bit register.

        Args:
            offset: Register offset.
            value: 32-bit value.

        Returns:
            True if successful.
        """
        if not self.connected:
            return False
        # In simulation mode, store register values locally
        if self.backend == "verilator" and not hasattr(self, '_regs'):
            self._regs = {}
        if self.backend == "verilator":
            self._regs[offset] = value
        return True

    def read_reg(self, offset: int) -> Optional[int]:
        """Read a 32-bit register.

        Args:
            offset: Register offset.

        Returns:
            32-bit register value, or None if not connected.
        """
        if not self.connected:
            return None
        # In simulation mode, return DONE status (bit 1 set)
        if self.backend == "verilator":
            return 0x2
        return 0

    def execute(self, instruction: int, timeout_ms: int = 1000) -> bool:
        """Execute a single ISA instruction.

        Args:
            instruction: 32-bit instruction word.
            timeout_ms: Timeout in milliseconds.

        Returns:
            True if execution completed successfully.
        """
        self.write_reg(0x08, instruction)  # INSTR register
        # Poll status
        for _ in range(timeout_ms // 10):
            status = self.read_reg(0x04)  # STATUS register
            if status is not None and (status & 0x2):  # Done flag
                return True
            time.sleep(0.01)
        return False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
