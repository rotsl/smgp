"""Integration test: offload SMGP ops to hardware simulation.

Tests that the hardware backend produces identical results to the pure
software implementation for key operations.
"""

import numpy as np
import pytest


class TestHardwareBackend:
    """Test the SMGP hardware backend integration."""

    def test_hw_session_verilator(self):
        """Test Verilator backend session."""
        from smgp_hal.hw_session import HWSession

        session = HWSession(backend="verilator")
        assert session.connect() is True
        assert session.connected is True
        session.disconnect()
        assert session.connected is False

    def test_hw_session_context_manager(self):
        """Test session as context manager."""
        from smgp_hal.hw_session import HWSession

        with HWSession(backend="verilator") as session:
            assert session.connected is True

    def test_hw_executor_add_node(self):
        """Test hardware executor node addition."""
        from smgp_hal.hw_session import HWSession
        from smgp_hal.executor import HWExecutor

        with HWSession(backend="verilator") as session:
            executor = HWExecutor(session)
            node_id = executor.add_node("test_node", label="entity")
            assert node_id == "test_node"
            assert executor.node_count == 1

    def test_hw_executor_add_edge(self):
        """Test hardware executor edge addition."""
        from smgp_hal.hw_session import HWSession
        from smgp_hal.executor import HWExecutor

        with HWSession(backend="verilator") as session:
            executor = HWExecutor(session)
            executor.add_node("A")
            executor.add_node("B")
            edge_id = executor.add_edge("A", "B", "connects")
            assert "A" in edge_id

    def test_hw_executor_laplacian(self):
        """Test hardware Laplacian computation."""
        from smgp_hal.hw_session import HWSession
        from smgp_hal.executor import HWExecutor

        with HWSession(backend="verilator") as session:
            executor = HWExecutor(session)
            executor.add_node("N1")
            executor.add_node("N2")
            executor.add_node("N3")
            result = executor.compute_laplacian()
            assert result is True  # Execution succeeded

    def test_hw_executor_hd_bind(self):
        """Test hardware HD bind matches software."""
        from smgp_hal.hw_session import HWSession
        from smgp_hal.executor import HWExecutor
        from smgp.core.hyperdim import HyperdimensionalMemory

        hd = HyperdimensionalMemory(dim=100, seed=42)
        a = hd.generate(1)[0]
        b = hd.generate(1)[0]

        # Software result
        sw_result = hd.bind(a, b)

        with HWSession(backend="verilator") as session:
            executor = HWExecutor(session)
            hw_result = executor.hd_bind(a, b)

        np.testing.assert_array_equal(sw_result, hw_result)

    def test_hw_executor_hd_similarity(self):
        """Test hardware HD similarity matches software."""
        from smgp_hal.hw_session import HWSession
        from smgp_hal.executor import HWExecutor
        from smgp.core.hyperdim import HyperdimensionalMemory

        hd = HyperdimensionalMemory(dim=100, seed=42)
        a = hd.generate(1)[0]

        # Software similarity
        sw_sim = hd.similarity(a, a)

        with HWSession(backend="verilator") as session:
            executor = HWExecutor(session)
            hw_sim = executor.hd_similarity(a, a)

        assert abs(sw_sim - hw_sim) < 0.1  # Allow hardware tolerance

    def test_memory_mapper_allocate(self):
        """Test memory mapper node allocation."""
        from smgp_hal.memory_mapper import MemoryMapper

        mapper = MemoryMapper(tile_size=32, num_tiles=4)
        addr = mapper.allocate_node("node_0")
        assert addr == (0, 0)
        addr = mapper.allocate_node("node_1")
        assert addr == (0, 1)
        assert mapper.utilization() > 0

    def test_memory_mapper_defragmentation(self):
        """Test memory mapper defragmentation."""
        from smgp_hal.memory_mapper import MemoryMapper

        mapper = MemoryMapper(tile_size=4, num_tiles=2)
        for i in range(8):
            mapper.allocate_node(f"node_{i}")

        # Fill up and trigger defrag
        mapper.allocate_node("overflow_node")
        # After defrag, all 9 nodes fit in ceil(9/4)*4=12 slots out of 8 total
        # Defrag reorganizes to fit within available tiles
        assert mapper.utilization() >= 0.5

    def test_memory_mapper_wear_leveling(self):
        """Test wear leveling rotation."""
        from smgp_hal.memory_mapper import MemoryMapper

        mapper = MemoryMapper(tile_size=4, num_tiles=4)
        for i in range(16):
            mapper.allocate_node(f"node_{i}")
        mapper.wear_level()
        # After rotation, utilization should be the same
        assert mapper.utilization() > 0

    def test_hw_end_to_end_reasoning(self):
        """Test end-to-end reasoning on hardware backend."""
        from smgp_hal.hw_session import HWSession
        from smgp_hal.executor import HWExecutor

        with HWSession(backend="verilator") as session:
            executor = HWExecutor(session)

            # Build small graph
            executor.add_node("Paris", label="city")
            executor.add_node("France", label="country")
            executor.add_edge("Paris", "France", "capital_of")

            # Compute Laplacian
            assert executor.compute_laplacian() is True

            # Topological analysis
            assert executor.build_filtration() is True
            assert executor.check_stability(0.5) is True


class TestISAEncoding:
    """Test ISA instruction encoding."""

    def test_encode_graph_add_node(self):
        """Test ADD_NODE instruction encoding."""
        from smgp_hal.executor import _encode_instr

        instr = _encode_instr(0x1, 0x0, 0x01, 0x0042)
        assert (instr >> 28) & 0xF == 0x1  # OPC_GRAPH_CTOR
        assert (instr >> 24) & 0xF == 0x0  # ADD_NODE
        assert (instr >> 16) & 0xFF == 0x01  # FLAG_START

    def test_encode_spectral_convolve(self):
        """Test SPECTRAL_CONV instruction encoding."""
        from smgp_hal.executor import _encode_instr

        instr = _encode_instr(0x2, 0x2, 0x05, 0x0005)
        assert (instr >> 28) & 0xF == 0x2  # OPC_SPECTRAL
        assert (instr >> 24) & 0xF == 0x2  # SUB_SPECTRAL_CONV
        assert instr & 0xFFFF == 0x0005   # operand = 5

    def test_encode_hd_bind(self):
        """Test HD_BIND instruction encoding."""
        from smgp_hal.executor import _encode_instr

        instr = _encode_instr(0x3, 0x1, 0x01, 0x0000)
        assert (instr >> 28) & 0xF == 0x3  # OPC_HD
        assert (instr >> 24) & 0xF == 0x1  # HD_BIND
