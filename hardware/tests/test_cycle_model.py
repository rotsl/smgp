"""
Tests for the Cycle-Accurate Hardware Simulator Python Interface.

Validates that CycleSimulator can be imported, its interface is correct,
and that ImportError is raised when the shared library is not available.
"""

import os
import sys
import platform
import pytest

# Ensure the hardware package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.sw.smgp_hal.cycle_simulator import CycleSimulator


class TestCycleSimulatorImport:
    """Verify the module can be imported and the class interface is correct."""

    def test_class_exists(self):
        """CycleSimulator class should be importable."""
        assert CycleSimulator is not None

    def test_opcode_constants(self):
        """Expected opcode constants must be defined."""
        assert CycleSimulator.OP_NOP == 0x00
        assert CycleSimulator.OP_COMPUTE_LAPLACIAN == 0x01
        assert CycleSimulator.OP_EIGEN_DECOMPOSITION == 0x02
        assert CycleSimulator.OP_HD_BIND == 0x03
        assert CycleSimulator.OP_GRAPH_REWRITE == 0x04
        assert CycleSimulator.OP_DMA_READ == 0x80
        assert CycleSimulator.OP_DMA_WRITE == 0x81

    def test_constructor(self):
        """CycleSimulator should be constructable without a library path."""
        sim = CycleSimulator()
        assert sim is not None
        assert sim._lib is None
        assert sim._cycle_count == 0
        assert sim._is_open is False

    def test_import_error_without_so(self):
        """Entering context manager without .so should raise ImportError."""
        sim = CycleSimulator(lib_path="/nonexistent/libfoo.so")
        with pytest.raises(ImportError, match="Could not load"):
            with sim:
                pass  # pragma: no cover

    def test_cycle_count_initial(self):
        """Initial cycle count should be zero."""
        sim = CycleSimulator()
        assert sim.get_cycle_count() == 0

    def test_reset(self):
        """Reset should set cycle count to zero."""
        sim = CycleSimulator()
        sim._cycle_count = 42
        sim.reset()
        assert sim.get_cycle_count() == 0

    def test_is_available_false(self):
        """is_available should be False without a loaded library."""
        sim = CycleSimulator()
        assert sim.is_available is False

    def test_opcode_name(self):
        """_opcode_name should return human-readable names."""
        assert CycleSimulator._opcode_name(0x00) == "NOP"
        assert CycleSimulator._opcode_name(0x01) == "COMPUTE_LAPLACIAN"
        assert CycleSimulator._opcode_name(0x03) == "HD_BIND"
        assert CycleSimulator._opcode_name(0xFF) == "UNKNOWN_0xFF"

    def test_estimate_cycles(self):
        """Software cycle estimator should match the SystemC model."""
        # Laplacian: nnz cycles
        assert CycleSimulator._estimate_cycles(0x01, 0, 1024) == 1024
        # Eigen: operand passes k*nnz
        assert CycleSimulator._estimate_cycles(0x02, 0, 16384) == 16384
        # HD bind: 1 cycle
        assert CycleSimulator._estimate_cycles(0x03, 0, 0) == 1
        # DMA read: operand + 2
        assert CycleSimulator._estimate_cycles(0x80, 0, 256) == 258
        # NOP: 1 cycle
        assert CycleSimulator._estimate_cycles(0x00, 0, 0) == 1

    def test_execute_without_context_raises(self):
        """Calling execute_instruction outside context manager should raise."""
        sim = CycleSimulator()
        with pytest.raises(RuntimeError, match="not open"):
            sim.execute_instruction(0x01, 0, 1024)


class TestCycleSimulatorWithMockLib:
    """Tests using a manual work-around: use the software estimator directly.

    Since we may not have the .so, we patch _is_open to exercise the
    execute_instruction path.
    """

    def test_execute_sequence(self):
        """Run a sequence of instructions and check cumulative cycles."""
        sim = CycleSimulator()
        # Manually enable the simulator for testing
        sim._is_open = True

        # Instruction 1: Laplacian (nnz=1024)
        r1 = sim.execute_instruction(0x01, 0, 1024)
        assert r1["cycles"] == 1024
        assert r1["opcode_name"] == "COMPUTE_LAPLACIAN"
        assert r1["status"] == "OK"

        # Instruction 2: Eigen-decomp (k*nnz = 16*1024 = 16384)
        r2 = sim.execute_instruction(0x02, 0, 16384)
        assert r2["cycles"] == 16384
        assert sim.get_cycle_count() == 1024 + 16384

        # Instruction 3: HD Bind
        r3 = sim.execute_instruction(0x03, 0, 0)
        assert r3["cycles"] == 1
        assert sim.get_cycle_count() == 1024 + 16384 + 1

        # Instruction 4: NOP
        r4 = sim.execute_instruction(0x00, 0, 0)
        assert r4["cycles"] == 1

        total = sim.get_cycle_count()
        assert total == 1024 + 16384 + 1 + 1  # 17410

        sim.close()

    def test_result_dict_keys(self):
        """Result dictionary should contain all expected keys."""
        sim = CycleSimulator()
        sim._is_open = True
        result = sim.execute_instruction(0x01, 0, 100)

        expected_keys = {
            "opcode", "opcode_name", "sub_opcode", "operand",
            "cycles", "cumulative_cycles", "status",
        }
        assert set(result.keys()) == expected_keys
        sim.close()

    def test_context_manager_exit(self):
        """__exit__ should set _is_open to False."""
        sim = CycleSimulator()
        sim._is_open = True
        sim._lib = None
        sim.close()
        assert sim._is_open is False
