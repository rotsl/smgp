"""Tests for the HyperdimensionalMemory module."""
import numpy as np
import pytest

from smgp.core.hyperdim import HyperdimensionalMemory


@pytest.fixture
def hd():
    return HyperdimensionalMemory(dim=1000, seed=42)


class TestHyperdimensionalMemory:
    def test_generate(self, hd):
        vecs = hd.generate(5)
        assert vecs.shape == (5, 1000)
        assert set(np.unique(vecs)) <= {-1, 1}

    def test_generate_single(self, hd):
        vec = hd.generate(1)
        assert vec.shape == (1, 1000)

    def test_bundle(self, hd):
        vecs = hd.generate(5)
        bundled = hd.bundle(vecs)
        assert bundled.shape == (1000,)
        assert set(np.unique(bundled)) <= {-1, 1}

    def test_bundle_single(self, hd):
        vec = hd.generate(1)[0]
        bundled = hd.bundle(vec)
        assert bundled.shape == (1000,)
        np.testing.assert_array_equal(bundled, vec)

    def test_bind_unbind_self_inverse(self, hd):
        a = hd.generate(1)[0]
        b = hd.generate(1)[0]
        bound = hd.bind(a, b)
        unbound = hd.unbind(bound, b)
        # For bipolar vectors, bind/unbind should approximately recover a
        # (exact for element-wise multiplication of bipolar vectors)
        np.testing.assert_array_equal(unbound, a)

    def test_permute(self, hd):
        v = hd.generate(1)[0]
        p = hd.permute(v, shifts=1)
        assert p.shape == (1000,)
        # Cyclic shift: last element of v should be first of permuted
        assert p[0] == v[-1]
        assert p[1] == v[0]

    def test_permute_multiple_shifts(self, hd):
        v = hd.generate(1)[0]
        p1 = hd.permute(v, shifts=1)
        p2 = hd.permute(p1, shifts=1)
        assert p2[0] == p1[-1]
        assert p2[1] == p1[0]

    def test_similarity_self(self, hd):
        v = hd.generate(1)[0]
        sim = hd.similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-10)

    def test_similarity_different(self, hd):
        a = hd.generate(1)[0]
        b = hd.generate(1)[0]
        sim = hd.similarity(a, b)
        # Random bipolar vectors should have low similarity
        assert abs(sim) < 0.2  # Very likely for D=1000

    def test_most_similar(self, hd):
        query = hd.generate(1)[0]
        memory = hd.generate(20)
        idx, sims = hd.most_similar(query, memory, k=5)
        assert len(idx) == 5
        assert len(sims) == 5
        # Similarities should be sorted descending
        assert all(sims[i] >= sims[i+1] for i in range(len(sims) - 1))

    def test_most_similar_with_self(self, hd):
        query = hd.generate(1)[0]
        memory = hd.generate(9)
        # Insert query into memory
        memory[3] = query
        idx, sims = hd.most_similar(query, memory, k=1)
        assert idx[0] == 3
        assert sims[0] == pytest.approx(1.0, abs=1e-10)
