"""Hyper-parameter auto-tuner for SMGP configurations.

:class:`Tuner` searches over a user-supplied parameter space and selects
the configuration that maximises a user-supplied metric function.

Currently supported methods:
  - ``"grid"`` — exhaustive grid search.

The metric function receives a config dict (which is also used to build
an :class:`~smgp.config.SMGPCConfig` internally) and returns a float
where **higher is better**.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from smgp.config import SMGPConfig

if TYPE_CHECKING:
    from smgp.core.graph import SpectralMemoryGraph


class Tuner:
    """Hyper-parameter tuner for SMGP.

    Parameters
    ----------
    graph : SpectralMemoryGraph
        The graph used as context for metric evaluation.
    metric_fn : callable
        ``metric_fn(config_dict) -> float``.  Must return a scalar where
        higher means better.
    param_space : dict
        Mapping of parameter name → list of values to try.

    Example
    -------
    ::

        def my_metric(cfg):
            return cfg["hd_dim"] * 0.001

        tuner = Tuner(graph, my_metric, {"hd_dim": [100, 200]})
        best = tuner.optimize(method="grid")
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        metric_fn: Callable[[dict[str, Any]], float],
        param_space: dict[str, list[Any]],
    ) -> None:
        from smgp.core.graph import SpectralMemoryGraph as _SMG  # noqa: F401, N814

        self.graph: SpectralMemoryGraph = graph
        self.metric_fn = metric_fn
        self.param_space = param_space
        self._best_config: dict[str, Any] | None = None
        self._best_score: float = float("-inf")
        self._history: list[dict[str, Any]] = []

    def optimize(
        self,
        method: str = "grid",
        n_iter: int = 10,
    ) -> dict[str, Any]:
        """Run hyper-parameter search and return the best configuration.

        Parameters
        ----------
        method : str
            Search method.  Only ``"grid"`` is implemented.
        n_iter : int
            Maximum iterations (only used for non-grid methods).

        Returns
        -------
        dict
            Best configuration found.

        Raises
        ------
        ValueError
            If *method* is not supported.
        """
        if method == "grid":
            return self._grid_search()
        else:
            raise ValueError(f"Unknown optimization method: {method!r}")

    def _grid_search(self) -> dict[str, Any]:
        """Exhaustive grid search over :attr:`param_space`.

        Returns
        -------
        dict
            Best configuration.
        """
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        if not keys:
            return {}

        best_score = float("-inf")
        best_config: dict[str, Any] = {}

        for combination in itertools.product(*values):
            config_dict = dict(zip(keys, combination))
            try:
                # Validate / build SMGPConfig to catch bad params early
                SMGPConfig(**{
                    k: v for k, v in config_dict.items()
                    if k in SMGPConfig.__dataclass_fields__
                })
            except (TypeError, ValueError):
                continue  # Skip invalid combinations

            score = self.metric_fn(config_dict)
            self._history.append({
                "config": config_dict,
                "score": score,
            })

            if score > best_score:
                best_score = score
                best_config = config_dict.copy()

        self._best_config = best_config
        self._best_score = best_score
        return best_config

    def save_best(self, path: str) -> None:
        """Save the best configuration as a JSON file.

        Parameters
        ----------
        path : str
            Destination file path.

        Raises
        ------
        RuntimeError
            If :meth:`optimize` has not been called yet.
        """
        if self._best_config is None:
            raise RuntimeError("Call optimize() before save_best().")

        # Build an SMGPConfig from the best dict and save it
        valid_kwargs = {
            k: v for k, v in self._best_config.items()
            if k in SMGPConfig.__dataclass_fields__
        }
        config = SMGPConfig(**valid_kwargs)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        config.save(out_path)

    @property
    def best_score(self) -> float:
        """Score of the best configuration found so far."""
        return self._best_score

    @property
    def history(self) -> list[dict[str, Any]]:
        """Full history of evaluated configurations and their scores."""
        return list(self._history)
