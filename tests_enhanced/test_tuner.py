"""Tests for :class:`Tuner`.

Creates a tiny graph, defines a trivial metric function, and verifies
that grid search selects the configuration with the highest metric.
"""
from __future__ import annotations

import json

import pytest

from smgp.core.graph import SpectralMemoryGraph
from smgp.utils.tuner import Tuner


class TestTuner:
    """Tests for the SMGP hyper-parameter tuner."""

    @pytest.fixture()
    def graph(self) -> SpectralMemoryGraph:
        return SpectralMemoryGraph(hd_dim=100, seed=0)

    def test_grid_search_returns_best(self, graph: SpectralMemoryGraph) -> None:
        """Grid search should return the config with the highest metric."""

        def metric(cfg):
            return cfg["hd_dim"] * 2

        tuner = Tuner(
            graph,
            metric_fn=metric,
            param_space={"hd_dim": [100, 200, 300]},
        )

        best = tuner.optimize(method="grid")
        assert best == {"hd_dim": 300}
        assert tuner.best_score == 600.0

    def test_grid_search_history(self, graph: SpectralMemoryGraph) -> None:
        """History should record every evaluated combination."""

        def metric(cfg):
            return cfg["hd_dim"]

        tuner = Tuner(
            graph,
            metric_fn=metric,
            param_space={"hd_dim": [10, 20]},
        )
        tuner.optimize(method="grid")

        assert len(tuner.history) == 2
        scores = {h["score"] for h in tuner.history}
        assert scores == {10.0, 20.0}

    def test_save_best(self, graph: SpectralMemoryGraph, tmp_path) -> None:
        """save_best should write a valid JSON config file."""

        def metric(cfg):
            return cfg["hd_dim"]

        tuner = Tuner(
            graph,
            metric_fn=metric,
            param_space={"hd_dim": [100, 200]},
        )
        tuner.optimize(method="grid")

        out = tmp_path / "best_config.json"
        tuner.save_best(str(out))

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["hd_dim"] == 200

    def test_save_best_before_optimize_raises(
        self, graph: SpectralMemoryGraph, tmp_path
    ) -> None:
        """Calling save_best before optimize should raise RuntimeError."""
        tuner = Tuner(
            graph,
            metric_fn=lambda cfg: 0.0,
            param_space={"hd_dim": [100]},
        )
        with pytest.raises(RuntimeError, match="Call optimize"):
            tuner.save_best(str(tmp_path / "x.json"))

    def test_invalid_method_raises(self, graph: SpectralMemoryGraph) -> None:
        """An unknown optimization method should raise ValueError."""
        tuner = Tuner(
            graph,
            metric_fn=lambda cfg: 0.0,
            param_space={},
        )
        with pytest.raises(ValueError, match="Unknown optimization method"):
            tuner.optimize(method="random")

    def test_empty_param_space(self, graph: SpectralMemoryGraph) -> None:
        """An empty param space should return an empty dict."""
        tuner = Tuner(
            graph,
            metric_fn=lambda cfg: 42.0,
            param_space={},
        )
        best = tuner.optimize(method="grid")
        assert best == {}

    def test_multi_param_search(self, graph: SpectralMemoryGraph) -> None:
        """Grid search over multiple parameters should test all combos."""

        def metric(cfg):
            return cfg["hd_dim"] + cfg["num_heads"] * 10

        tuner = Tuner(
            graph,
            metric_fn=metric,
            param_space={
                "hd_dim": [100, 200],
                "num_heads": [2, 4],
            },
        )
        best = tuner.optimize(method="grid")
        assert best == {"hd_dim": 200, "num_heads": 4}
        assert len(tuner.history) == 4  # 2 x 2 grid
