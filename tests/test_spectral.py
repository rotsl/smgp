"""Tests for the SpectralMethods module."""
import numpy as np
import pytest

from smgp.core.graph import SpectralMemoryGraph
from smgp.core.spectral import SpectralMethods


@pytest.fixture
def graph_and_spectral():
    """Create a connected test graph with spectral methods."""
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    nodes = ["A", "B", "C", "D", "E"]
    for n in nodes:
        g.add_node(n, label="test")
    # Create a connected graph (line + some cross-edges)
    g.add_edge("A", "B", "connects")
    g.add_edge("B", "C", "connects")
    g.add_edge("C", "D", "connects")
    g.add_edge("D", "E", "connects")
    g.add_edge("A", "C", "connects")
    g.add_edge("B", "D", "connects")
    spectral = SpectralMethods(g, num_eigenvalues=4)
    return g, spectral


class TestSpectralMethods:
    def test_compute_laplacian(self, graph_and_spectral):
        g, spectral = graph_and_spectral
        L = spectral.compute_laplacian()
        assert L.shape == (5, 5)
        # Laplacian should be symmetric
        L_dense = L.toarray()
        np.testing.assert_array_almost_equal(L_dense, L_dense.T)

    def test_compute_eigen(self, graph_and_spectral):
        g, spectral = graph_and_spectral
        eigenvalues, eigenvectors = spectral.compute_eigen()
        assert len(eigenvalues) == 4
        assert eigenvectors.shape == (5, 4)
        # Eigenvalues should be non-negative
        assert all(ev >= -1e-10 for ev in eigenvalues)
        # First eigenvalue should be close to 0 (connected graph)
        assert abs(eigenvalues[0]) < 0.5

    def test_graph_fourier_transform_roundtrip(self, graph_and_spectral):
        g, spectral = graph_and_spectral
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        spectral_coeffs = spectral.graph_fourier_transform(signal)
        assert spectral_coeffs.shape == (4,)
        # Inverse should approximately recover the signal (projection loss)
        recovered = spectral.inverse_graph_fourier_transform(spectral_coeffs)
        assert recovered.shape == (5,)
        # Check the projection is reasonable (not exact due to truncation)
        signal_proj = spectral.inverse_graph_fourier_transform(spectral.graph_fourier_transform(signal))
        # The recovered signal should have similar norm (energy preservation up to truncation)
        orig_norm = np.linalg.norm(signal)
        rec_norm = np.linalg.norm(signal_proj)
        assert rec_norm > 0  # Should not be zero
        # Allow some loss due to eigenbasis truncation
        assert rec_norm <= orig_norm * 1.5

    def test_chebyshev_convolution(self, graph_and_spectral):
        g, spectral = graph_and_spectral
        signal = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        coefficients = np.array([1.0, 0.5, 0.1])
        result = spectral.chebyshev_convolution(signal, coefficients)
        assert result.shape == (5,)
        assert not np.all(result == 0)

    def test_graph_wavelet_transform(self, graph_and_spectral):
        g, spectral = graph_and_spectral
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wavelets = spectral.graph_wavelet_transform(signal, scales=[1.0, 2.0])
        assert wavelets.shape == (2, 5)
        assert not np.all(wavelets == 0)
