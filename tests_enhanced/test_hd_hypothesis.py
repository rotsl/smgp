"""Property-based tests for Hyperdimensional Memory using Hypothesis.

Uses hypothesis to generate random inputs and verify algebraic properties
of HD vector operations (bind/unbind, bundle, permute, similarity).

References:
  - Hypothesis: https://hypothesis.readthedocs.io/
  - Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation, 1(2).
"""
from __future__ import annotations

import numpy as np
import pytest

from smgp.core.hyperdim import HyperdimensionalMemory

HAS_HYPOTHESIS = False
try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    pass

# Use a smaller dim for fast property-based testing
_HD_DIM = 1000


def _make_hd():
    """Create a fresh HD engine."""
    return HyperdimensionalMemory(dim=_HD_DIM, seed=42)


# Shared health check suppression: 1000-dim vectors are "large" but necessary
_SUPPRESS = [HealthCheck.function_scoped_fixture, HealthCheck.large_base_example] if HAS_HYPOTHESIS else []


def _hd_vector():
    """Generate strategy for HD bipolar vectors of fixed dimension."""
    return st.just(_HD_DIM).flatmap(
        lambda dim: st.builds(
            lambda bits: (2 * bits.astype(np.int8) - 1).astype(np.int8),
            st.integers(min_value=0, max_value=1).filter(lambda x: True).map(
                lambda _: np.random.default_rng(0).choice(
                    np.array([-1, 1], dtype=np.int8), size=dim
                )
            )
            if False else st.none()  # Use simpler approach below
        )
    )


# Simpler: use numpy arrays strategy
def _hd_arrays():
    """Generate HD bipolar vectors via np arrays strategy."""
    return st.builds(
        lambda x: (2 * (x > 0).astype(np.int8) - 1).astype(np.int8),
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=_HD_DIM, max_size=_HD_DIM,
        ).map(lambda lst: np.array(lst, dtype=np.float64))
    )


if HAS_HYPOTHESIS:

    class TestBindUnbindConsistency:
        """Test that bind/unbind is self-inverse."""

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(a=_hd_arrays(), b=_hd_arrays())
        def test_bind_unbind_consistency(self, a, b):
            hd = _make_hd()
            a = a.astype(np.int8)
            b = b.astype(np.int8)
            bound = hd.bind(a, b)
            recovered = hd.unbind(bound, b)
            assert np.array_equal(recovered, a)
            sim = hd.similarity(recovered, a)
            assert sim > 0.99

    class TestBundlePreservesSimilarity:
        """Test that bundling preserves some similarity to components."""

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(n=st.integers(min_value=2, max_value=8))
        def test_bundle_preserves_similarity(self, n):
            hd = _make_hd()
            vectors = hd.generate(n)
            bundled = hd.bundle(vectors)
            min_threshold = max(0.05, 0.5 / (n ** 0.5))
            for i in range(n):
                sim = hd.similarity(bundled, vectors[i])
                assert sim > min_threshold

    class TestPermuteInversibility:
        """Test that permutation by k then by -k returns the original vector."""

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(v=_hd_arrays(), k=st.integers(min_value=-_HD_DIM, max_value=_HD_DIM))
        def test_permute_inversibility(self, v, k):
            hd = _make_hd()
            v = v.astype(np.int8)
            permuted = hd.permute(v, shifts=k)
            recovered = hd.permute(permuted, shifts=-k)
            assert np.array_equal(recovered, v)

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(v=_hd_arrays())
        def test_permute_zero_shift_is_identity(self, v):
            hd = _make_hd()
            v = v.astype(np.int8)
            result = hd.permute(v, shifts=0)
            assert np.array_equal(result, v)

    class TestSimilarityRange:
        """Test that similarity is always in [-1, 1]."""

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(a=_hd_arrays(), b=_hd_arrays())
        def test_similarity_range(self, a, b):
            hd = _make_hd()
            sim = hd.similarity(a.astype(np.int8), b.astype(np.int8))
            assert -1.0 <= sim <= 1.0

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(a=_hd_arrays())
        def test_similarity_self_is_one(self, a):
            hd = _make_hd()
            sim = hd.similarity(a.astype(np.int8), a.astype(np.int8))
            assert abs(sim - 1.0) < 1e-10

        @settings(max_examples=20, suppress_health_check=_SUPPRESS)
        @given(a=_hd_arrays())
        def test_similarity_negated_is_negative_one(self, a):
            hd = _make_hd()
            a = a.astype(np.int8)
            neg_a = -a
            sim = hd.similarity(a, neg_a)
            assert abs(sim - (-1.0)) < 1e-10

    class TestBundleSingleVector:
        """Edge case: bundling a single vector should return it unchanged."""

        @settings(max_examples=10, suppress_health_check=_SUPPRESS)
        @given(v=_hd_arrays())
        def test_bundle_single_vector(self, v):
            hd = _make_hd()
            vectors = v.astype(np.int8).reshape(1, -1)
            bundled = hd.bundle(vectors)
            assert np.array_equal(bundled, v.astype(np.int8))

else:
    class TestBindUnbindConsistency:
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_bind_unbind_consistency(self): pass

    class TestBundlePreservesSimilarity:
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_bundle_preserves_similarity(self): pass

    class TestPermuteInversibility:
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_permute_inversibility(self): pass
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_permute_zero_shift_is_identity(self): pass

    class TestSimilarityRange:
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_similarity_range(self): pass
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_similarity_self_is_one(self): pass
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_similarity_negated_is_negative_one(self): pass

    class TestBundleSingleVector:
        @pytest.mark.skip(reason="hypothesis not installed")
        def test_bundle_single_vector(self): pass
