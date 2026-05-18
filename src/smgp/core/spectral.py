"""Spectral graph transforms, wavelets, and Laplacian decompositions.

Implements the spectral theory of graphs as described in:
- Chung, F.R.K. (1997). "Spectral Graph Theory." CBMS Regional Conference Series, No. 92.
- Shuman, D.I., et al. (2013). "The Emerging Field of Signal Processing on Graphs."
  IEEE Signal Processing Magazine, 30(3), 83–98.
- Hammond, D.K., et al. (2011). "Wavelets on Graphs via Spectral Graph Theory."
  Applied and Computational Harmonic Analysis, 30(2), 129–150.

Key concepts:
  - Graph Laplacian L = D - A (combinatorial) or L = I - D^{-1/2} A D^{-1/2} (normalized).
  - Graph Fourier Transform: hat(f) = U^T f, where U are Laplacian eigenvectors.
  - Chebyshev polynomial approximation: enables O(K|E|) spectral filtering.
  - Graph wavelets: multi-scale spectral filters at multiple frequency scales.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from smgp.core.graph import SpectralMemoryGraph

try:
    from smgp.hw_bridge import HardwareBridge as _HardwareBridge
except ImportError:
    _HardwareBridge = None  # type: ignore[assignment,misc]


class SpectralMethods:
    """Spectral analysis engine for knowledge graphs.

    Provides eigen-decomposition of the normalized graph Laplacian, Graph
    Fourier Transform and its inverse, Chebyshev polynomial spectral
    convolution, and graph wavelet transforms for multi-scale analysis.

    Attributes:
        graph: The knowledge graph to analyze.
        num_eigenvalues: Number of eigenvalues/eigenvectors to compute.
        _laplacian: Cached normalized Laplacian matrix.
        _eigenvalues: Cached eigenvalues.
        _eigenvectors: Cached eigenvectors.

    References:
        Chung, F.R.K. (1997). "Spectral Graph Theory."
        Shuman, D.I., et al. (2013). "Signal Processing on Graphs." IEEE SPM.
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        num_eigenvalues: int = 50,
        executor: Any = None,
    ) -> None:
        """Initialize spectral methods.

        Args:
            graph: Knowledge graph to analyze.
            num_eigenvalues: Number of eigenvalues to compute. Should be less
                than the number of nodes in the graph.
            executor: Optional hardware executor.  Inherits from *graph* when
                ``None`` and the graph carries one.  Defaults to ``None``.
        """
        self.graph = graph
        self.num_eigenvalues = min(num_eigenvalues, graph.num_nodes)
        self._laplacian: sparse.csr_matrix | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self.executor = executor or getattr(graph, "executor", None)
        self._bridge = (
            _HardwareBridge(self.executor)
            if self.executor is not None and _HardwareBridge is not None
            else None
        )

    def compute_laplacian(self) -> sparse.csr_matrix:
        """Compute the normalized graph Laplacian.

        The normalized Laplacian is defined as:
            L_norm = I - D^{-1/2} A D^{-1/2}

        where D is the degree matrix and A is the adjacency matrix. The
        eigenvalues of L_norm lie in [0, 2], which provides nice numerical
        properties for spectral filtering.

        Returns:
            scipy.sparse.csr_matrix of shape (N, N).

        References:
            Chung, F.R.K. (1997). "Spectral Graph Theory," Ch. 1-2.
        """
        if self._laplacian is not None:
            return self._laplacian

        # Offload to hardware if available
        if self._bridge is not None:
            try:
                adj = self.graph.adjacency_matrix()
                result = self._bridge.compute_laplacian(
                    adj.toarray() if sparse.issparse(adj) else adj, normalized=True
                )
                self._laplacian = sparse.csr_matrix(result)
                return self._laplacian
            except NotImplementedError:
                pass

        adj = self.graph.adjacency_matrix()
        n = adj.shape[0]

        if n == 0:
            self._laplacian = sparse.csr_matrix((0, 0))
            return self._laplacian

        # Symmetrize and binarize the adjacency matrix
        adj = adj + adj.T
        adj.data[:] = 1  # Binary

        # Degree vector
        degrees = np.array(adj.sum(axis=1)).flatten()
        # Handle isolated nodes (degree 0)
        degrees[degrees == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(degrees)

        # D^{-1/2} A D^{-1/2}
        d_inv_sqrt_sparse = sparse.diags(d_inv_sqrt)
        identity = sparse.eye(n, format="csr")
        L = identity - d_inv_sqrt_sparse @ adj @ d_inv_sqrt_sparse
        # Enforce exact symmetry to prevent floating-point drift that would
        # break the GFT↔IGFT roundtrip (eigenvectors of a non-symmetric matrix
        # are not unitary, so U @ U^T ≠ I).
        L = (L + L.T) * 0.5
        self._laplacian = L.tocsr()

        return self._laplacian

    def compute_eigen(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the eigen-decomposition of the normalized Laplacian.

        Uses scipy.sparse.linalg.eigsh for efficient computation of the
        smallest eigenvalues, which correspond to the lowest-frequency
        graph Fourier modes.

        Returns:
            Tuple of (eigenvalues, eigenvectors) where:
              - eigenvalues: np.ndarray of shape (K,)
              - eigenvectors: np.ndarray of shape (N, K)

        Raises:
            ValueError: If the graph has fewer than 2 nodes.

        References:
            Chung, F.R.K. (1997). "Spectral Graph Theory," Theorem 1.2.
        """
        if self._eigenvalues is not None and self._eigenvectors is not None:
            return self._eigenvalues, self._eigenvectors

        # Offload to hardware if available
        if self._bridge is not None:
            try:
                L = self.compute_laplacian()
                L_dense = L.toarray() if sparse.issparse(L) else L
                evals, evecs = self._bridge.compute_eigen(L_dense, self.num_eigenvalues)
                self._eigenvalues = evals
                self._eigenvectors = evecs
                return self._eigenvalues, self._eigenvectors
            except NotImplementedError:
                pass

        L = self.compute_laplacian()
        n = L.shape[0]

        if n < 2:
            raise ValueError(
                "Graph must have at least 2 nodes for eigen-decomposition."
            )

        k = min(self.num_eigenvalues, n)

        if k >= n:
            # Full basis: dense eigh gives a complete orthonormal set so that
            # U @ U^T = I (exact GFT↔IGFT roundtrip, error < 1e-10).
            L_dense = L.toarray() if sparse.issparse(L) else np.asarray(L, dtype=float)
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        else:
            try:
                # eigsh requires k < n
                eigenvalues, eigenvectors = eigsh(L, k=k, which="SM")
                # Re-orthonormalize via thin QR to correct floating-point drift
                # in the eigenvector matrix (U^T U = I within machine precision).
                eigenvectors, _ = np.linalg.qr(eigenvectors)
            except Exception:
                # Fallback: dense eigh for small or ill-conditioned graphs
                L_dense = L.toarray() if sparse.issparse(L) else np.asarray(L, dtype=float)
                eigenvalues_all, eigenvectors_all = np.linalg.eigh(L_dense)
                eigenvalues = eigenvalues_all[:k]
                eigenvectors = eigenvectors_all[:, :k]

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors[:, idx]
        return self._eigenvalues, self._eigenvectors

    def graph_fourier_transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply the Graph Fourier Transform.

        hat(f) = U^T f

        where U is the matrix of eigenvectors (graph Fourier basis).

        Args:
            signal: Node-wise signal of shape (N,).

        Returns:
            Spectral coefficients of shape (K,) where K = num_eigenvalues.

        References:
            Shuman, D.I., et al. (2013). "Signal Processing on Graphs." IEEE SPM.
        """
        _, eigenvectors = self.compute_eigen()
        return eigenvectors.T @ signal

    def inverse_graph_fourier_transform(
        self, spectral_signal: np.ndarray
    ) -> np.ndarray:
        """Apply the inverse Graph Fourier Transform.

        f = U hat(f)

        Args:
            spectral_signal: Spectral coefficients of shape (K,).

        Returns:
            Reconstructed signal of shape (N,).

        References:
            Shuman, D.I., et al. (2013). "Signal Processing on Graphs." IEEE SPM.
        """
        _, eigenvectors = self.compute_eigen()
        return eigenvectors @ spectral_signal

    def chebyshev_convolution(
        self,
        signal: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """Fast spectral convolution via Chebyshev polynomial approximation.

        Approximates the spectral filter g(Lambda) ≈ sum_{k=0}^{K-1} theta_k T_k(L_tilde)
        where L_tilde = 2L/lambda_max - I is the scaled Laplacian and T_k
        are Chebyshev polynomials of the first kind.

        This avoids explicit eigen-decomposition and runs in O(K|E|) time.

        Args:
            signal: Node-wise input signal of shape (N,).
            coefficients: Filter coefficients theta_k of shape (K,).

        Returns:
            Filtered signal of shape (N,).

        References:
            Defferrard, M., et al. (2016). "Convolutional Neural Networks on
                Graphs with Fast Localized Spectral Filtering." NeurIPS.
        """
        L = self.compute_laplacian()
        n = L.shape[0]
        K = len(coefficients)

        # Scale Laplacian: L_tilde = 2L/lambda_max - I
        if self._eigenvalues is not None and len(self._eigenvalues) > 0:
            lambda_max = max(self._eigenvalues[-1], 1e-6)
        else:
            # Estimate lambda_max
            try:
                lambda_max_val = eigsh(
                    L, k=1, which="LM", return_eigenvectors=False
                )
                lambda_max = (
                    float(lambda_max_val[0])
                    if len(lambda_max_val) > 0
                    else 2.0
                )
            except Exception:
                lambda_max = 2.0

        lambda_max = max(lambda_max, 1e-6)
        L_scaled = (2.0 * L / lambda_max) - sparse.eye(n, format="csr")

        # Chebyshev recursion:
        # T_0(x) = 1, T_1(x) = x, T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
        T_prev = sparse.eye(n, format="csr")  # T_0
        result = coefficients[0] * (T_prev @ signal)

        if K > 1:
            T_current = L_scaled  # T_1
            result = result + coefficients[1] * (T_current @ signal)

            for k in range(2, K):
                T_next = 2.0 * L_scaled @ T_current - T_prev
                T_next = T_next.tocsr()
                result = result + coefficients[k] * (T_next @ signal)
                T_prev = T_current
                T_current = T_next

        return np.asarray(result).flatten()

    def graph_wavelet_transform(
        self,
        signal: np.ndarray,
        scales: list[float] | None = None,
    ) -> np.ndarray:
        """Compute the graph wavelet transform at multiple scales.

        Graph wavelets are defined as:
            psi_s(Lambda) = g(s * Lambda)
        where g is a spectral kernel (e.g., Mexican hat) and s is the scale.

        We approximate each scale's filter via Chebyshev polynomials.

        Args:
            signal: Node-wise input signal of shape (N,).
            scales: List of wavelet scales. Default [1.0, 2.0, 4.0].

        Returns:
            Wavelet coefficients of shape (num_scales, N).

        References:
            Hammond, D.K., et al. (2011). "Wavelets on Graphs via Spectral
                Graph Theory." ACHA, 30(2), 129–150.
        """
        if scales is None:
            scales = [1.0, 2.0, 4.0]

        n = signal.shape[0]
        num_scales = len(scales)
        wavelet_coeffs = np.zeros((num_scales, n))

        for i, s in enumerate(scales):
            # Heat kernel-based wavelet: g_s(lambda) = exp(-s * lambda)
            # Approximate with Chebyshev coefficients
            cheb_order = 5
            cheb_coeffs = self._heat_kernel_cheb_coefficients(s, cheb_order)
            wavelet_coeffs[i] = self.chebyshev_convolution(signal, cheb_coeffs)

        return wavelet_coeffs

    def _heat_kernel_cheb_coefficients(
        self, scale: float, order: int
    ) -> np.ndarray:
        """Compute Chebyshev coefficients for the heat kernel g(lambda) = exp(-s * lambda).

        Uses Chebyshev polynomial expansion of the heat kernel evaluated
        on [-1, 1] (the domain of the scaled Laplacian).

        Args:
            scale: Heat kernel scale parameter s.
            order: Number of Chebyshev terms K.

        Returns:
            Chebyshev coefficients of shape (K,).
        """
        # Sample the heat kernel on Chebyshev points
        num_samples = max(order * 4, 64)
        cheb_points = np.cos(
            np.pi * (np.arange(num_samples) + 0.5) / num_samples
        )
        # Map from [-1, 1] to [0, 2] (eigenvalue range of normalized Laplacian)
        eigenval_points = cheb_points + 1.0  # linearly mapped to [0, 2]
        kernel_values = np.exp(-scale * eigenval_points)

        # Discrete cosine transform to get Chebyshev coefficients
        coeffs = np.zeros(order)
        sample_indices = np.arange(num_samples)
        for k in range(order):
            coeffs[k] = (2.0 / num_samples) * np.sum(
                kernel_values
                * np.cos(k * np.pi * (sample_indices + 0.5) / num_samples)
            )
        coeffs[0] /= 2.0  # Correct the DC term
        return coeffs
