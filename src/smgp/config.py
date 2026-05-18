"""Global configuration for SMGP components.

Loads settings from YAML/JSON config files and environment variables with
the SMGP_ prefix. All components respect these settings.

References:
    - Dellamonica et al. (2020). "Configurable AI Systems" — design patterns for
      hierarchical configuration management in ML pipelines.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HardwareConfig:
    """Hardware acceleration settings.

    Attributes:
        enabled: Whether to use the FPGA/simulator accelerator.
        device: Path to the PCIe device node (e.g. ``/dev/smgpu0``).
        fallback: If ``True``, fall back to pure-Python when the device is
            unavailable rather than raising an error.
    """

    enabled: bool = False
    device: str = "/dev/smgpu0"
    fallback: bool = True


@dataclass
class SMGPConfig:
    """Master configuration for the SMGP system.

    Attributes:
        hd_dim: Dimensionality of hyperdimensional vectors. Default 10000 as per
            the binary sparse distributed representation literature (Kanerva, 1988).
        hidden_dim: Dimension of internal attention embeddings.
        num_heads: Number of attention heads for spectral attention.
        num_scales: Number of scales in the graph wavelet pyramid.
        max_eigenvalues: Maximum number of eigenvalues for spectral decomposition.
        memory_capacity: Maximum number of items in the memory store.
        pruning_threshold: Wasserstein distance threshold for topological stability.
        seed: Random seed for reproducibility.
        device: Compute device ('cpu' or 'cuda').

    References:
        Kanerva, P. (1988). "Sparse Distributed Memory". MIT Press.
        Chung, F.R.K. (1997). "Spectral Graph Theory". CBMS Regional Conference Series.
    """
    hd_dim: int = 10000
    hidden_dim: int = 256
    num_heads: int = 8
    num_scales: int = 3
    max_eigenvalues: int = 64
    memory_capacity: int = 100000
    pruning_threshold: float = 0.5
    seed: int | None = None
    device: str = "cpu"
    # Memory lifecycle settings
    forgetting_policy: str = "low_persistence"  # "low_persistence", "lru", "random"
    forgetting_rate: float = 0.1
    # Spectral settings
    chebyshev_order: int = 5
    wavelet_scales: list = field(default_factory=lambda: [1.0, 2.0, 4.0])
    # Hardware acceleration
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SMGPConfig:
        """Load configuration from a YAML or JSON file.

        Args:
            path: Path to the configuration file.

        Returns:
            SMGPConfig instance with values from the file.
        """
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> SMGPConfig:
        """Load configuration from a dictionary.

        Handles nested ``hardware`` sub-dict by converting it into a
        :class:`HardwareConfig` instance automatically.

        Args:
            data: Flat or nested configuration dictionary.

        Returns:
            SMGPConfig instance.
        """
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "hardware" in filtered and isinstance(filtered["hardware"], dict):
            filtered["hardware"] = HardwareConfig(**filtered["hardware"])
        return cls(**filtered)

    @classmethod
    def from_env(cls) -> SMGPConfig:
        """Load configuration from environment variables with SMGP_ prefix.

        Example: SMGP_HD_DIM=20000 sets hd_dim to 20000.

        Returns:
            SMGPConfig instance with values from environment variables.
        """
        kwargs: dict[str, Any] = {}
        env_map = {
            "SMGP_HD_DIM": ("hd_dim", int),
            "SMGP_HIDDEN_DIM": ("hidden_dim", int),
            "SMGP_NUM_HEADS": ("num_heads", int),
            "SMGP_NUM_SCALES": ("num_scales", int),
            "SMGP_MAX_EIGENVALUES": ("max_eigenvalues", int),
            "SMGP_MEMORY_CAPACITY": ("memory_capacity", int),
            "SMGP_PRUNING_THRESHOLD": ("pruning_threshold", float),
            "SMGP_SEED": ("seed", int),
            "SMGP_DEVICE": ("device", str),
            "SMGP_FORGETTING_POLICY": ("forgetting_policy", str),
            "SMGP_FORGETTING_RATE": ("forgetting_rate", float),
            "SMGP_CHEBYSHEV_ORDER": ("chebyshev_order", int),
        }
        for env_var, (field_name, field_type) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                kwargs[field_name] = field_type(val)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_executor_from_config(config: SMGPConfig) -> Any | None:
    """Create an HWExecutor from a config, or return None if disabled.

    Args:
        config: SMGPConfig with a populated ``hardware`` field.

    Returns:
        An ``HWExecutor`` instance if ``config.hardware.enabled`` is ``True``,
        otherwise ``None``.

    Raises:
        ImportError: If ``hardware.sw.smgp_hal`` is not available and hardware
            is enabled.
    """
    if not config.hardware.enabled:
        return None
    from hardware.sw.smgp_hal.executor import HWExecutor  # noqa: PLC0415
    from hardware.sw.smgp_hal.hw_session import HWSession  # noqa: PLC0415

    session = HWSession(
        device=config.hardware.device,
        fallback=config.hardware.fallback,
    )
    return HWExecutor(session=session)
