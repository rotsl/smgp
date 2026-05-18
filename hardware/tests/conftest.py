"""Pytest fixtures for hardware integration tests."""

import pytest
from smgp_hal.hw_session import HWSession


@pytest.fixture(scope="session")
def hw_session():
    """Provide a Verilator-backed hardware session for all tests."""
    session = HWSession(backend="verilator")
    session.connect()
    yield session
    session.disconnect()


@pytest.fixture
def hw_executor(hw_session):
    """Provide a hardware executor for a test."""
    from smgp_hal.executor import HWExecutor
    return HWExecutor(hw_session)


@pytest.fixture
def memory_mapper():
    """Provide a fresh memory mapper for each test."""
    from smgp_hal.memory_mapper import MemoryMapper
    return MemoryMapper(tile_size=64, num_tiles=8)
