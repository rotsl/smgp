"""Golden reference model for HD engine operations.

Replicates the hardware's bipolar vector operations in Python.
"""
import numpy as np
from typing import List, Tuple


class GoldenHDModel:
    """Golden reference for hyperdimensional vector operations.

    Matches the hardware HD engine's operations exactly:
    - Generate: random bipolar vectors
    - Bundle: majority vote (sign of sum)
    - Bind/Unbind: XOR (self-inverse)
    - Permute: cyclic shift
    - Similarity: dot product normalized by dimension
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def generate(self, n: int = 1) -> np.ndarray:
        """Generate n random bipolar vectors {+1, -1}^D."""
        return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, self.dim))

    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """Bundle via element-wise majority vote: sign(sum)."""
        if vectors.ndim == 1:
            return vectors
        sums = vectors.sum(axis=0)
        result = np.sign(sums).astype(np.int8)
        result[result == 0] = 1
        return result

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind: element-wise XOR (self-inverse for bipolar)."""
        return np.where(a == b, 1, -1).astype(np.int8)

    def unbind(self, bound: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Unbind (same as bind for bipolar)."""
        return self.bind(bound, b)

    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Cyclic left shift."""
        return np.roll(v, -shifts)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (equals dot product / D for bipolar)."""
        return float(np.dot(a.astype(np.float64), b.astype(np.float64))) / self.dim

    def to_bit_storage(self, v: np.ndarray) -> List[int]:
        """Convert from {-1,+1} to {0,1} storage format (matches hardware)."""
        return [0 if x == 1 else 1 for x in v]

    def from_bit_storage(self, bits: List[int]) -> np.ndarray:
        """Convert from {0,1} storage to {-1,+1} representation."""
        return np.array([1 if b == 0 else -1 for b in bits], dtype=np.int8)


if __name__ == "__main__":
    model = GoldenHDModel(dim=10000)

    a = model.generate(1)[0]
    b = model.generate(1)[0]

    bound = model.bind(a, b)
    unbound = model.unbind(bound, b)

    print(f"Bind/Unbind recovery: {np.array_equal(a, unbound)}")
    print(f"Similarity(a, a): {model.similarity(a, a):.6f}")
    print(f"Similarity(a, b): {model.similarity(a, b):.6f}")

    vectors = model.generate(5)
    bundled = model.bundle(vectors)
    print(f"Bundle shape: {bundled.shape}")
