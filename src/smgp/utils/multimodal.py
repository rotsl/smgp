"""Multi-modal graph node helpers for SMGP.

Provides convenience functions that encode images, text, and audio into
vector embeddings using popular models (CLIP, Whisper, etc.) and store
the results as node properties on a
:class:`~smgp.core.graph.SpectralMemoryGraph`.

All heavy-weight imports (**PIL**, **transformers**, **torch**) are
**lazy** — they are only imported when the function is actually called.
If the required package is missing an :class:`ImportError` with a
helpful message is raised.

References
----------
  - Radford, A., et al. (2021). "Learning Transferable Visual Models."
    ICML.
  - Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale
    Weak Supervision." arXiv.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smgp.core.graph import SpectralMemoryGraph


def _require_packages(
    packages: dict[str, str],
) -> None:
    """Check that all *packages* are importable; raise otherwise.

    Parameters
    ----------
    packages : dict
        Mapping of ``import_name`` → ``pip_name``.
    """
    missing: list[str] = []
    for import_name, pip_name in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise ImportError(
            f"Missing required package(s): {', '.join(missing)}.  "
            "Install them with:  pip install " + " ".join(missing)
        )


def add_image_node(
    graph: SpectralMemoryGraph,
    node_id: str,
    image_path: str,
    model: str = "clip",
) -> str:
    """Add an image node with a CLIP visual embedding.

    Parameters
    ----------
    graph : SpectralMemoryGraph
        Target graph.
    node_id : str
        Unique node identifier.
    image_path : str
        Path to the image file.
    model : str
        Embedding model name (currently only ``"clip"`` supported).

    Returns
    -------
    str
        The *node_id* that was added.

    Raises
    ------
    ImportError
        If **PIL**, **torch**, or **transformers** are not installed.
    """
    if model != "clip":
        raise ValueError(f"Unsupported image model: {model!r} (use 'clip')")

    _require_packages({
        "PIL": "Pillow",
        "torch": "torch",
        "transformers": "transformers",
    })

    import torch  # type: ignore[import-untyped]
    from PIL import Image  # type: ignore[import-untyped]
    from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-untyped]

    # Load model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)

    embedding_np = embedding.squeeze(0).cpu().numpy().tolist()

    graph.add_node(
        node_id,
        label=image_path,
        properties={
            "embedding": embedding_np,
            "model": model,
            "modality": "image",
        },
    )
    return node_id


def add_text_node(
    graph: SpectralMemoryGraph,
    node_id: str,
    text: str,
    model: str = "clip",
) -> str:
    """Add a text node with a CLIP text embedding.

    Parameters
    ----------
    graph : SpectralMemoryGraph
        Target graph.
    node_id : str
        Unique node identifier.
    text : str
        Text content to embed.
    model : str
        Embedding model name (currently only ``"clip"`` supported).

    Returns
    -------
    str
        The *node_id* that was added.

    Raises
    ------
    ImportError
        If **torch** or **transformers** are not installed.
    """
    if model != "clip":
        raise ValueError(f"Unsupported text model: {model!r} (use 'clip')")

    _require_packages({
        "torch": "torch",
        "transformers": "transformers",
    })

    import torch  # type: ignore[import-untyped]
    from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-untyped]

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)

    embedding_np = embedding.squeeze(0).cpu().numpy().tolist()

    graph.add_node(
        node_id,
        label=text[:128],
        properties={
            "embedding": embedding_np,
            "model": model,
            "modality": "text",
        },
    )
    return node_id


def add_audio_node(
    graph: SpectralMemoryGraph,
    node_id: str,
    audio_path: str,
    model: str = "whisper",
) -> str:
    """Add an audio node with a placeholder embedding.

    .. note::
        This is currently a **placeholder**.  It stores metadata but does
        **not** yet produce a real embedding.  The API is kept for
        forward-compatibility when Whisper integration is added.

    Parameters
    ----------
    graph : SpectralMemoryGraph
        Target graph.
    node_id : str
        Unique node identifier.
    audio_path : str
        Path to the audio file.
    model : str
        Embedding model name (reserved for future use).

    Returns
    -------
    str
        The *node_id* that was added.

    Raises
    ------
    NotImplementedError
        Always — audio embedding is not yet implemented.
    """
    raise NotImplementedError(
        "Audio embedding via Whisper is not yet implemented.  "
        "Contributions are welcome."
    )
