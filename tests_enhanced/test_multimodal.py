"""Tests for multi-modal graph node helpers.

Uses ``unittest.mock`` to patch heavy-weight libraries (PIL, torch,
transformers) so the tests run without downloading real models.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smgp.core.graph import SpectralMemoryGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_mock_torch_transformers() -> None:
    """Inject mock torch/transformers/PIL into sys.modules for the duration."""
    for mod_name in ("torch", "torch.nn", "transformers", "PIL", "PIL.Image"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


def _make_clip_result(dim: int = 512) -> list:
    """Create a realistic-looking CLIP embedding list."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((dim,)).tolist()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddImageNode:
    """Tests for :func:`smgp.utils.multimodal.add_image_node`."""

    def test_image_node_gets_embedding(self) -> None:
        """After add_image_node the graph should contain an embedding property."""
        # Patch importlib so _require_packages never fails
        with patch("smgp.utils.multimodal.importlib.import_module", return_value=MagicMock()):
            # Reload the module to pick up fresh sys.modules state
            import importlib as _il

            import smgp.utils.multimodal as _mm
            _il.reload(_mm)

            # Now mock the CLIP model to return a known embedding
            fake_embedding_list = _make_clip_result(512)

            mock_torch_mod = MagicMock()
            mock_torch_mod.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_torch_mod.no_grad.return_value.__exit__ = MagicMock(return_value=False)

            mock_processor = MagicMock()
            mock_processor.return_value = {"pixel_values": np.zeros((1, 3))}

            mock_model = MagicMock()
            # Build the chain: .get_image_features(**inputs).squeeze(0).cpu().numpy().tolist()
            mock_numpy_obj = MagicMock()
            mock_numpy_obj.tolist.return_value = fake_embedding_list
            mock_model.get_image_features.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = mock_numpy_obj

            mock_clip_module = MagicMock()
            mock_clip_module.CLIPProcessor.from_pretrained.return_value = mock_processor
            mock_clip_module.CLIPModel.from_pretrained.return_value = mock_model

            mock_pil_module = MagicMock()
            mock_img = MagicMock()
            mock_pil_module.Image.open.return_value.convert.return_value = mock_img

            with (
                patch.dict(sys.modules, {
                    "torch": mock_torch_mod,
                    "transformers": mock_clip_module,
                    "PIL": mock_pil_module,
                    "PIL.Image": mock_pil_module.Image,
                }),
            ):
                graph = SpectralMemoryGraph(hd_dim=100, seed=0)
                _mm.add_image_node(graph, "img_1", "/fake/path.png", model="clip")

            node = graph.get_node("img_1")
            assert node is not None
            assert "embedding" in node["properties"]
            assert isinstance(node["properties"]["embedding"], list)
            assert len(node["properties"]["embedding"]) == 512
            assert node["properties"]["model"] == "clip"
            assert node["properties"]["modality"] == "image"


class TestAddTextNode:
    """Tests for :func:`smgp.utils.multimodal.add_text_node`."""

    def test_text_node_gets_embedding(self) -> None:
        """After add_text_node the graph should contain an embedding property."""
        with patch("smgp.utils.multimodal.importlib.import_module", return_value=MagicMock()):
            import importlib as _il

            import smgp.utils.multimodal as _mm
            _il.reload(_mm)

            fake_embedding_list = _make_clip_result(512)

            mock_torch_mod = MagicMock()
            mock_torch_mod.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_torch_mod.no_grad.return_value.__exit__ = MagicMock(return_value=False)

            mock_processor = MagicMock()
            mock_processor.return_value = {"input_ids": np.array([[1, 2, 3]])}

            mock_model = MagicMock()
            mock_numpy_obj = MagicMock()
            mock_numpy_obj.tolist.return_value = fake_embedding_list
            mock_model.get_text_features.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = mock_numpy_obj

            mock_clip_module = MagicMock()
            mock_clip_module.CLIPProcessor.from_pretrained.return_value = mock_processor
            mock_clip_module.CLIPModel.from_pretrained.return_value = mock_model

            with (
                patch.dict(sys.modules, {
                    "torch": mock_torch_mod,
                    "transformers": mock_clip_module,
                }),
            ):
                graph = SpectralMemoryGraph(hd_dim=100, seed=0)
                _mm.add_text_node(graph, "txt_1", "hello world", model="clip")

            node = graph.get_node("txt_1")
            assert node is not None
            assert "embedding" in node["properties"]
            assert isinstance(node["properties"]["embedding"], list)
            assert len(node["properties"]["embedding"]) == 512
            assert node["properties"]["model"] == "clip"
            assert node["properties"]["modality"] == "text"


class TestAddAudioNode:
    """Tests for :func:`smgp.utils.multimodal.add_audio_node`."""

    def test_audio_raises_not_implemented(self) -> None:
        """Audio embedding is a placeholder and should raise."""
        graph = SpectralMemoryGraph(hd_dim=100, seed=0)
        from smgp.utils.multimodal import add_audio_node

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            add_audio_node(graph, "aud_1", "/fake/audio.wav", model="whisper")
