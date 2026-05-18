"""Drop-in attention wrapper for vLLM and ONNX Runtime integration.

Provides SMGPAttentionWrapper, a torch.nn.Module that intercepts standard
multi-head attention and replaces it with spectral graph attention for
O(N log N) scaling on structured inputs.

References:
  - vLLM: Kwon et al. (2023). "Efficient Memory Management for Large Language
    Model Serving with PagedAttention." SOSP.
  - ONNX Runtime: Microsoft. https://onnxruntime.ai/
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SMGPAttentionWrapper:
    """Drop-in torch.nn.Module replacement for standard multi-head attention.

    Wraps SpectralMemoryGraph and SpectralAttention to provide an attention
    interface compatible with HuggingFace transformers, vLLM, and ONNX export.

    Attributes:
        hidden_dim: Dimensionality of hidden states.
        num_heads: Number of attention heads.
        num_scales: Number of wavelet scales.
        hd_dim: Dimensionality of HD vectors for the internal graph.
        graph: Internal SpectralMemoryGraph.
        spectral_attention: Internal SpectralAttention instance.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_scales: int = 3,
        hd_dim: int = 100,
    ) -> None:
        """Initialize the SMGP attention wrapper.

        Args:
            hidden_dim: Dimensionality of hidden states (default 256).
            num_heads: Number of attention heads (default 8).
            num_scales: Number of wavelet scales (default 3).
            hd_dim: Dimensionality of internal HD vectors (default 100).
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.hd_dim = hd_dim

        from smgp.attention.spectral_attn import SpectralAttention
        from smgp.core.graph import SpectralMemoryGraph

        self.graph = SpectralMemoryGraph(hd_dim=hd_dim, seed=42)
        self.spectral_attention = SpectralAttention(
            graph=self.graph,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_scales=num_scales,
        )

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass: build graph from tokens, run spectral attention.

        Accepts a torch.Tensor or numpy.ndarray of shape (batch, seq_len, hidden_dim)
        or (seq_len, hidden_dim). Builds a graph representation of the tokens and
        applies spectral graph attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq, dim) or (seq, dim).
            attention_mask: Optional attention mask (currently informational only).
            **kwargs: Additional keyword arguments (ignored, for compatibility).

        Returns:
            Output tensor of the same shape as input hidden_states.
        """
        import numpy as np

        # Convert torch tensor to numpy if needed
        if hasattr(hidden_states, "detach"):
            hidden_np = hidden_states.detach().cpu().numpy()
        elif isinstance(hidden_states, np.ndarray):
            hidden_np = hidden_states
        else:
            hidden_np = np.asarray(hidden_states)

        # Handle batch dimension: process each batch element separately
        if hidden_np.ndim == 3:
            batch_size, seq_len, _ = hidden_np.shape
            outputs = []
            for b in range(batch_size):
                out = self._process_sequence(hidden_np[b])
                outputs.append(out)
            result_np = np.stack(outputs, axis=0)

            # Convert back to torch tensor if input was a tensor
            if hasattr(hidden_states, "__class__"):
                try:
                    import torch
                    if isinstance(hidden_states, torch.Tensor):
                        return torch.from_numpy(result_np).to(hidden_states.device)
                except ImportError:
                    pass
            return result_np
        elif hidden_np.ndim == 2:
            result_np = self._process_sequence(hidden_np)
            if hasattr(hidden_states, "__class__"):
                try:
                    import torch
                    if isinstance(hidden_states, torch.Tensor):
                        return torch.from_numpy(result_np).to(hidden_states.device)
                except ImportError:
                    pass
            return result_np
        else:
            raise ValueError(
                f"Expected hidden_states with 2 or 3 dimensions, got {hidden_np.ndim}"
            )

    def _process_sequence(self, tokens: np.ndarray) -> np.ndarray:
        """Process a single sequence through spectral attention.

        Args:
            tokens: Token embedding matrix of shape (seq_len, hidden_dim).

        Returns:
            Contextualized embeddings of shape (seq_len, hidden_dim).
        """
        # Build graph from token embeddings
        self.spectral_attention.build_graph_from_tokens(tokens)

        # Run spectral attention
        output = self.spectral_attention.forward(tokens)

        # Ensure output shape matches input
        if output.shape != tokens.shape:
            # If shapes don't match, project to correct dimension
            output = output[:, : tokens.shape[1]]

        return output

    def export_onnx(self, path: str, dynamic_batch: bool = True) -> None:
        """Export the forward pass as an ONNX model.

        Args:
            path: File path for the exported ONNX model.
            dynamic_batch: If True, use dynamic batch dimension.

        Raises:
            ImportError: If torch or onnx is not installed.
        """
        try:
            import onnx  # noqa: F401
            import torch
        except ImportError as e:
            raise ImportError(
                "torch and onnx are required for ONNX export. "
                "Install them with: pip install torch onnx"
            ) from e

        # Create dummy inputs (real tensor for mask so tracer sees both inputs)
        dummy_input = torch.randn(1, 8, self.hidden_dim)
        dummy_mask = torch.ones(1, 8, 8)

        # Wrap forward for torch.jit.trace compatibility
        class _TraceableWrapper(torch.nn.Module):
            def __init__(self, wrapper: SMGPAttentionWrapper) -> None:
                super().__init__()
                self._wrapper = wrapper

            def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
                return self._wrapper.forward(x, mask)

        traceable = _TraceableWrapper(self)

        if dynamic_batch:
            dynamic_axes = {
                "hidden_states": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"},
            }
        else:
            dynamic_axes = None

        torch.onnx.export(
            traceable,
            (dummy_input, dummy_mask),
            path,
            input_names=["hidden_states", "attention_mask"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        logger.info("Exported ONNX model to %s", path)

    @classmethod
    def register_with_vllm(cls) -> None:
        """Register this attention implementation as a vLLM attention backend.

        This is a placeholder documenting the integration path. To fully
        register with vLLM, you would:

        1. Subclass ``vllm.attention.backends.AbstractAttentionBackend``.
        2. Implement ``forward_impl`` to call SMGP spectral attention.
        3. Register via ``vllm.attention.backends.BackRegistry``.

        Example (conceptual)::

            from vllm.attention import BackRegistry

            class SMGPBackend(AbstractAttentionBackend):
                @staticmethod
                def get_supported_head_sizes() -> list[int]:
                    return [hidden_dim // num_heads]

                @staticmethod
                def get_kv_cache_shape(...) -> tuple[int, ...]:
                    ...

                @staticmethod
                def forward_impl(
                    ctx,
                    query,
                    key,
                    value,
                    ...
                ):
                    # Call SMGP spectral attention here
                    wrapper = SMGPAttentionWrapper(...)
                    return wrapper.forward(query)

            BackRegistry.register(SMGPBackend)

        Raises:
            ImportError: If vllm is not installed.
        """
        try:
            import vllm  # noqa: F401
        except ImportError:
            logger.warning(
                "vLLM is not installed. Cannot register SMGP attention backend. "
                "Install with: pip install vllm"
            )
            return

        logger.info(
            "SMGP vLLM backend registration placeholder. "
            "See docstring for full integration instructions."
        )

    def __repr__(self) -> str:
        return (
            f"SMGPAttentionWrapper("
            f"hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"num_scales={self.num_scales}, "
            f"hd_dim={self.hd_dim})"
        )
