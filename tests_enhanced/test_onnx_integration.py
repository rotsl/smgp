"""Tests for ONNX / vLLM attention wrapper integration."""
from __future__ import annotations

import sys

import numpy as np
import pytest

torch_available = pytest.mark.skipif(
    sys.version_info < (3, 8) or True,
    reason="Torch import test; module import is primary check",
)

HAS_TORCH = False
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    pass


class TestSMGPAttentionWrapperImport:
    """Test that SMGPAttentionWrapper can be imported."""

    def test_import_module(self):
        """Module should be importable regardless of torch availability."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        assert SMGPAttentionWrapper is not None

    def test_class_instantiation(self):
        """Wrapper should instantiate without torch installed."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=64,
            num_heads=4,
            num_scales=2,
            hd_dim=100,
        )
        assert wrapper.hidden_dim == 64
        assert wrapper.num_heads == 4
        assert wrapper.hd_dim == 100
        assert wrapper.graph is not None

    def test_repr(self):
        """Wrapper repr should be informative."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(hidden_dim=128, num_heads=8, hd_dim=200)
        r = repr(wrapper)
        assert "SMGPAttentionWrapper" in r
        assert "hidden_dim=128" in r
        assert "num_heads=8" in r


class TestSMGPAttentionWrapperForward:
    """Test forward pass with numpy inputs (always available)."""

    def test_forward_2d_numpy(self):
        """Forward pass with 2D numpy input should work."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=32,
            num_heads=4,
            num_scales=1,
            hd_dim=100,
        )
        x = np.random.randn(4, 32).astype(np.float32)
        output = wrapper.forward(x)
        assert isinstance(output, np.ndarray)
        assert output.shape == (4, 32)

    def test_forward_3d_numpy(self):
        """Forward pass with 3D numpy input (batch) should work."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=32,
            num_heads=4,
            num_scales=1,
            hd_dim=100,
        )
        x = np.random.randn(2, 4, 32).astype(np.float32)
        output = wrapper.forward(x)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 4, 32)

    def test_forward_accepts_attention_mask(self):
        """Forward pass should accept attention_mask without error."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=32,
            num_heads=4,
            num_scales=1,
            hd_dim=100,
        )
        x = np.random.randn(3, 32).astype(np.float32)
        mask = np.ones((3, 3), dtype=np.float32)
        output = wrapper.forward(x, attention_mask=mask)
        assert isinstance(output, np.ndarray)
        assert output.shape[0] == 3


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
class TestSMGPAttentionWrapperTorch:
    """Tests requiring torch to be installed."""

    def test_forward_torch_tensor(self):
        """Forward pass with torch tensor should return torch tensor."""
        import torch

        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=32,
            num_heads=4,
            num_scales=1,
            hd_dim=100,
        )
        x = torch.randn(4, 32)
        output = wrapper.forward(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 32)

    def test_forward_torch_batch(self):
        """Forward pass with batched torch tensor."""
        import torch

        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=32,
            num_heads=4,
            num_scales=1,
            hd_dim=100,
        )
        x = torch.randn(2, 6, 32)
        output = wrapper.forward(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 6, 32)

    def test_export_onnx(self, tmp_path):
        """ONNX export should produce a file."""

        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        wrapper = SMGPAttentionWrapper(
            hidden_dim=32,
            num_heads=4,
            num_scales=1,
            hd_dim=100,
        )
        onnx_path = str(tmp_path / "smgp_attention.onnx")
        wrapper.export_onnx(onnx_path, dynamic_batch=True)
        import os
        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 0

    def test_register_with_vllm_placeholder(self):
        """register_with_vllm should not crash."""
        from smgp.integration.onnx_vllm import SMGPAttentionWrapper
        SMGPAttentionWrapper.register_with_vllm()
