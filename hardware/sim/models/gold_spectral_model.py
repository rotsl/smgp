"""Golden reference model for spectral engine fixed-point arithmetic.

Replicates the hardware's fixed-point computations in Python for
bit-accurate comparison with RTL simulation outputs.

References:
    Kastner, R. et al. (2010). "Floating-Point to Fixed-Point Conversion."
"""
import numpy as np
from typing import Tuple


class FixedPointConfig:
    """Configuration matching hardware parameters."""
    def __init__(self, int_bits: int = 8, frac_bits: int = 24, total_bits: int = 32):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = total_bits
        self.scale = 1 << frac_bits
        self.max_val = (1 << (total_bits - 1)) - 1
        self.min_val = -(1 << (total_bits - 1))


class GoldenSpectralModel:
    """Golden reference model for spectral computations.

    Matches the hardware's fixed-point arithmetic exactly so that
    simulation outputs can be compared bit-for-bit.
    """

    def __init__(self, config: FixedPointConfig | None = None):
        self.cfg = config or FixedPointConfig()

    def to_fp(self, val: float) -> int:
        """Convert float to fixed-point representation."""
        scaled = int(round(val * self.cfg.scale))
        return max(min(scaled, self.cfg.max_val), self.cfg.min_val)

    def from_fp(self, fp_val: int) -> float:
        """Convert fixed-point to float."""
        return fp_val / self.cfg.scale

    def fp_mul(self, a: int, b: int) -> int:
        """Fixed-point multiplication (matches hardware fp_mul)."""
        product = (a * b) >> self.cfg.frac_bits
        return self._saturate(product)

    def fp_add(self, a: int, b: int) -> int:
        """Fixed-point addition with saturation."""
        result = a + b
        return self._saturate(result)

    def fp_sub(self, a: int, b: int) -> int:
        """Fixed-point subtraction."""
        return self.fp_add(a, -b)

    def fp_abs(self, a: int) -> int:
        """Absolute value."""
        return max(-a, a)

    def fp_reciprocal(self, x: int, iterations: int = 8) -> int:
        """Newton-Raphson reciprocal (matches hardware)."""
        two = self.to_fp(2.0)
        y = (~x + 1) >> 1  # Initial guess
        for _ in range(iterations):
            xy = self.fp_mul(x, y)
            two_minus_xy = self.fp_sub(two, xy)
            y = self.fp_mul(y, two_minus_xy)
        return y

    def fp_sqrt(self, x: int) -> int:
        """Integer square root for Euclidean distance."""
        if x <= 0:
            return 0
        val = x
        rem = 0
        root = 0
        for i in range(31, -1, -1):
            rem = (rem << 2) | ((val >> (2 * i)) & 3)
            trial = (root << 2) | 1
            if rem >= trial:
                rem -= trial
                root = (root << 1) | 1
            else:
                root <<= 1
        return root >> (self.cfg.frac_bits // 2)

    def compute_laplacian(
        self, adjacency: np.ndarray, degrees: np.ndarray
    ) -> np.ndarray:
        """Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2} in fixed-point."""
        n = adjacency.shape[0]
        laplacian = np.zeros((n, n), dtype=np.int64)
        identity_fp = self.to_fp(1.0)

        for i in range(n):
            for j in range(n):
                if degrees[i] > 0 and degrees[j] > 0:
                    d_inv_sqrt = self.fp_reciprocal(
                        self.to_fp(np.sqrt(degrees[i] * degrees[j]))
                    )
                    scaled = self.fp_mul(
                        int(adjacency[i, j]) << (self.cfg.frac_bits - 8),
                        d_inv_sqrt
                    )
                    laplacian[i, j] = -scaled
                if i == j:
                    laplacian[i, j] += identity_fp

        return laplacian

    def chebyshev_convolve(
        self, laplacian: np.ndarray, signal: np.ndarray, coefficients: list
    ) -> np.ndarray:
        """Chebyshev polynomial spectral convolution in fixed-point."""
        n = len(signal)
        result = np.zeros(n, dtype=np.int64)

        # T_0 = 1, T_1 = L, T_k = 2*L*T_{k-1} - T_{k-2}
        for i in range(n):
            result[i] = self.fp_mul(coefficients[0], self.to_fp(1.0))

        if len(coefficients) > 1:
            t_prev = np.zeros(n, dtype=np.int64)
            t_current = np.zeros(n, dtype=np.int64)
            for i in range(n):
                t_current[i] = laplacian[i][i] if laplacian[i][i] != 0 else 0
                result[i] = self.fp_add(result[i], self.fp_mul(coefficients[1], t_current[i]))

            for k in range(2, len(coefficients)):
                t_next = np.zeros(n, dtype=np.int64)
                for i in range(n):
                    double_l = self.fp_mul(self.to_fp(2.0), laplacian[i][i])
                    t_next[i] = self.fp_sub(
                        self.fp_mul(double_l, t_current[i]),
                        t_prev[i]
                    )
                    result[i] = self.fp_add(result[i], self.fp_mul(coefficients[k], t_next[i]))
                t_prev = t_current.copy()
                t_current = t_next.copy()

        return result

    def _saturate(self, val: int) -> int:
        """Saturate to fixed-point range."""
        return max(min(val, self.cfg.max_val), self.cfg.min_val)


if __name__ == "__main__":
    model = GoldenSpectralModel()

    # Test basic operations
    a = model.to_fp(1.5)
    b = model.to_fp(2.0)
    print(f"fp_mul(1.5, 2.0) = {model.from_fp(model.fp_mul(a, b))}")
    print(f"fp_add(1.5, 2.0) = {model.from_fp(model.fp_add(a, b))}")
    print(f"fp_sub(1.5, 2.0) = {model.from_fp(model.fp_sub(a, b))}")
    print(f"fp_reciprocal(2.0) = {model.from_fp(model.fp_reciprocal(b))}")

    # Test Laplacian
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    deg = np.array([2, 2, 2])
    lap = model.compute_laplacian(adj, deg)
    print(f"\nLaplacian:\n{lap}")
