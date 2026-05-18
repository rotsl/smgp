"""
PYNQ Overlay for SMGPU FPGA Accelerator

Provides a high-level Python interface for loading and controlling the SMGPU
bitstream on Xilinx Zynq / Zynq UltraScale+ platforms via the PYNQ framework.

Lazy-imports pynq — if the PYNQ runtime is not installed, a helpful
ImportError is raised at runtime (not at import time).
"""

from typing import List, Optional


class SMGPOverlay:
    """PYNQ overlay that loads the SMGPU bitstream and maps DMA buffers.

    Usage::

        try:
            overlay = SMGPOverlay("smgpu.bit")
            buf = overlay.allocate_buffer(4096)
            # ... interact with accelerator ...
            overlay.free_buffer(buf)
        except ImportError:
            print("PYNQ not available on this platform")
    """

    def __init__(self, bitstream_path: str = "smgpu.bit"):
        """Initialise the overlay.

        Parameters
        ----------
        bitstream_path:
            Path to the .bit or .xclbin file for the SMGPU overlay.

        Raises
        ------
        ImportError
            If the ``pynq`` package is not installed.
        FileNotFoundError
            If the bitstream file does not exist.
        """
        self._bitstream_path = bitstream_path
        self._pynq_overlay = None
        self._dma = None
        self._buffers: List = []
        self._loaded = False

        # Lazy import — only fail when actually constructing
        try:
            from pynq import Overlay, PL  # noqa: F401
            self._Overlay = Overlay
            self._PL = PL
        except ImportError:
            raise ImportError(
                "PYNQ runtime is not installed. "
                "SMGPOverlay requires the pynq package, which is only "
                "available on supported Xilinx Zynq platforms. "
                "\n\nInstall with: pip install pynq"
            )

        self.load_bitstream(bitstream_path)

    def load_bitstream(self, bitstream_path: str) -> None:
        """Load a bitstream onto the FPGA fabric.

        Parameters
        ----------
        bitstream_path:
            Absolute or relative path to the bitstream file.
        """
        import os
        if not os.path.isfile(bitstream_path):
            raise FileNotFoundError(
                f"Bitstream file not found: {bitstream_path}"
            )

        self._bitstream_path = bitstream_path
        self._pynq_overlay = self._Overlay(bitstream_path)

        # Try to locate the DMA engine in the overlay
        try:
            self._dma = self._pynq_overlay.axi_dma_0
        except AttributeError:
            # Try alternate naming conventions
            for attr_name in ("dma", "axi_dma", "axi_dma_0", "smgp_dma"):
                if hasattr(self._pynq_overlay, attr_name):
                    self._dma = getattr(self._pynq_overlay, attr_name)
                    break

        if self._dma is None:
            print("[SMGP] WARNING: No DMA engine found in overlay.")

        self._loaded = True
        print(f"[SMGP] Bitstream loaded: {bitstream_path}")

    def allocate_buffer(self, size: int, dtype=None) -> "numpy.ndarray":
        """Allocate a contiguous DMA buffer backed by physical memory.

        Parameters
        ----------
        size:
            Number of elements in the buffer.
        dtype:
            NumPy dtype (default: ``numpy.float32``).

        Returns
        -------
        numpy.ndarray
            A contiguous array suitable for DMA transfer.
        """
        if dtype is None:
            import numpy as np
            dtype = np.float32

        import numpy as np
        from pynq import allocate

        if self._dma is None:
            raise RuntimeError("No DMA engine available for buffer allocation.")

        buf = allocate(shape=(size,), dtype=dtype)
        self._buffers.append(buf)
        print(f"[SMGP] Allocated buffer: {size} elements ({buf.nbytes} bytes)")
        return buf

    def free_buffer(self, buffer) -> None:
        """Free a previously allocated DMA buffer.

        Parameters
        ----------
        buffer:
            The buffer handle returned by :meth:`allocate_buffer`.
        """
        if buffer in self._buffers:
            self._buffers.remove(buffer)

        # PYNQ buffers are freed by the garbage collector, but we can
        # explicitly release the underlying memory:
        if hasattr(buffer, "freebuffer"):
            buffer.freebuffer()

        print("[SMGP] Buffer freed.")

    def free_all_buffers(self) -> None:
        """Free all allocated DMA buffers."""
        for buf in list(self._buffers):
            self.free_buffer(buf)
        self._buffers.clear()

    @property
    def is_loaded(self) -> bool:
        """Return *True* if the bitstream is currently loaded."""
        return self._loaded

    @property
    def dma(self):
        """Return the underlying PYNQ DMA handle (or *None*)."""
        return self._dma

    @property
    def overlay(self):
        """Return the underlying PYNQ Overlay handle."""
        return self._pynq_overlay

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"SMGPOverlay(bitstream='{self._bitstream_path}', status={status})"

    def close(self) -> None:
        """Free buffers and release overlay resources."""
        self.free_all_buffers()
        self._pynq_overlay = None
        self._dma = None
        self._loaded = False

    def __enter__(self) -> "SMGPOverlay":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
