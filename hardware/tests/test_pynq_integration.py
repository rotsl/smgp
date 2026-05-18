"""
Tests for PYNQ Integration (SMGPOverlay and PYNQHAL)

Verifies that the PYNQ backend classes can be imported and that their
interfaces are correct.  Tests are skipped when the ``pynq`` package is
not available (non-Zynq environments).
"""

import os
import sys
import pytest

# Ensure the hardware package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Check PYNQ availability
# ---------------------------------------------------------------------------
PYNQ_AVAILABLE = False
try:
    # Verify the real pynq package (not our local stub) provides Overlay
    from pynq import Overlay  # noqa: F401
    PYNQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNQ_AVAILABLE = False

requires_pynq = pytest.mark.skipif(
    not PYNQ_AVAILABLE,
    reason="PYNQ runtime not available (requires Zynq platform)",
)


# ===========================================================================
# Unit tests that run everywhere (no PYNQ required)
# ===========================================================================

class TestPYNQHALImport:
    """Verify the PYNQHAL module can be imported without pynq installed."""

    def test_pynqhal_importable(self):
        """PYNQHAL class should be importable (lazy import)."""
        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        assert PYNQHAL is not None

    def test_pynqhal_opcode_constants(self):
        """Opcode constants should be defined and match ISA spec."""
        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        assert PYNQHAL.OP_NOP == 0x00
        assert PYNQHAL.OP_COMPUTE_LAPLACIAN == 0x01
        assert PYNQHAL.OP_EIGEN_DECOMPOSITION == 0x02
        assert PYNQHAL.OP_HD_BIND == 0x03
        assert PYNQHAL.OP_GRAPH_REWRITE == 0x04
        assert PYNQHAL.OP_DMA_READ == 0x80
        assert PYNQHAL.OP_DMA_WRITE == 0x81

    def test_pynqhal_raises_import_error_without_pynq(self):
        """Constructing PYNQHAL without pynq should raise ImportError."""
        if PYNQ_AVAILABLE:
            pytest.skip("PYNQ is installed — cannot test ImportError path")

        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        with pytest.raises(ImportError, match="PYNQ runtime is not installed"):
            PYNQHAL("smgpu.bit")

    def test_pynqhal_emulate_cycles(self):
        """Software cycle emulation should match the SystemC model."""
        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        assert PYNQHAL._emulate_cycles(0x01, 1024) == 1024
        assert PYNQHAL._emulate_cycles(0x02, 16384) == 16384
        assert PYNQHAL._emulate_cycles(0x03, 0) == 1
        assert PYNQHAL._emulate_cycles(0x00, 0) == 1
        assert PYNQHAL._emulate_cycles(0x80, 256) == 258

    def test_pynqhal_context_manager_protocol(self):
        """PYNQHAL should support the context manager protocol."""
        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        assert hasattr(PYNQHAL, "__enter__")
        assert hasattr(PYNQHAL, "__exit__")
        assert hasattr(PYNQHAL, "close")


class TestSMGPOverlayImport:
    """Verify the SMGPOverlay module can be imported without pynq installed."""

    def test_overlay_importable(self):
        """SMGPOverlay class should be importable (lazy import)."""
        from hardware.sw.pynq.smgp_overlay import SMGPOverlay
        assert SMGPOverlay is not None

    def test_overlay_raises_import_error_without_pynq(self):
        """Constructing SMGPOverlay without pynq should raise ImportError."""
        if PYNQ_AVAILABLE:
            pytest.skip("PYNQ is installed — cannot test ImportError path")

        from hardware.sw.pynq.smgp_overlay import SMGPOverlay
        with pytest.raises(ImportError, match="PYNQ runtime is not installed"):
            SMGPOverlay("smgpu.bit")

    def test_overlay_raises_filenotfound_for_missing_bitstream(self):
        """Constructing SMGPOverlay with non-existent bitstream should raise."""
        if PYNQ_AVAILABLE:
            pytest.skip("PYNQ is installed — testing requires real bitstream")

        from hardware.sw.pynq.smgp_overlay import SMGPOverlay
        # ImportError (pynq) should come before FileNotFoundError
        with pytest.raises(ImportError):
            SMGPOverlay("/nonexistent/smgpu.bit")

    def test_overlay_context_manager_protocol(self):
        """SMGPOverlay should support the context manager protocol."""
        from hardware.sw.pynq.smgp_overlay import SMGPOverlay
        assert hasattr(SMGPOverlay, "__enter__")
        assert hasattr(SMGPOverlay, "__exit__")
        assert hasattr(SMGPOverlay, "close")
        assert hasattr(SMGPOverlay, "is_loaded")


# ===========================================================================
# Integration tests that require PYNQ runtime
# ===========================================================================

@requires_pynq
class TestPYNQHALWithPYNQ:
    """Tests that run only when PYNQ is available."""

    def test_pynqhal_construction_fails_without_bitstream(self):
        """PYNQHAL should raise FileNotFoundError for missing bitstream."""
        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        with pytest.raises(FileNotFoundError):
            PYNQHAL("/nonexistent/smgpu.bit")

    def test_pynqhal_execute_result_keys(self):
        """Execute result dict should contain all expected keys."""
        from hardware.sw.pynq.smgp_pynq_hal import PYNQHAL
        expected_keys = {
            "opcode", "sub_opcode", "flags", "operand",
            "status", "cycles", "error",
        }
        # Can't execute without a real overlay, but we can check the
        # interface by mocking _is_open
        hal = PYNQHAL.__new__(PYNQHAL)
        hal._is_open = True
        hal._registers = None
        # This should use the emulation path
        result = hal.execute(0x01, 0, 0, 1024)
        assert set(result.keys()) == expected_keys
        assert result["status"] == "EMULATED"
        assert result["cycles"] == 1024


@requires_pynq
class TestSMGPOverlayWithPYNQ:
    """Tests that run only when PYNQ is available."""

    def test_overlay_construction_fails_without_bitstream(self):
        """SMGPOverlay should raise FileNotFoundError for missing bitstream."""
        from hardware.sw.pynq.smgp_overlay import SMGPOverlay
        with pytest.raises(FileNotFoundError):
            SMGPOverlay("/nonexistent/smgpu.bit")
