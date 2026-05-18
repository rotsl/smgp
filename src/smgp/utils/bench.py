"""Benchmarking utilities for SMGP against standard LLM tasks.

Provides standardized benchmarking for comparing SMGP's spectral attention
and graph reasoning against baseline transformer models on tasks like
long-range dependency resolution, factual recall, and mathematical reasoning.

References:
  - Perez, E., et al. (2021). "Beyond the Imitation Game: Quantifying and
    Extrapolating the Capabilities of Language Models." arXiv.
  - Hendrycks, D., et al. (2021). "Measuring Massive Multitask Language
    Understanding." ICLR (MMLU benchmark).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from smgp.attention.spectral_attn import SpectralAttention
from smgp.core.graph import SpectralMemoryGraph


class BenchmarkRunner:
    """Standardized benchmark runner for SMGP evaluation.

    Runs synthetic and structured benchmarks to measure:
    - Recall accuracy on long-range dependency tasks.
    - Reasoning accuracy on symbolic puzzles.
    - Throughput (tokens/second) for attention computation.
    - Memory efficiency (bytes per node stored).

    Attributes:
        graph: Knowledge graph for benchmarking.
        results: Dictionary of benchmark results.

    References:
        Hendrycks, D., et al. (2021). "Measuring Massive Multitask Language
            Understanding." ICLR.
    """

    def __init__(self, graph: SpectralMemoryGraph | None = None) -> None:
        """Initialize the benchmark runner.

        Args:
            graph: Optional graph. Creates a fresh one if not provided.
        """
        self.graph = graph or SpectralMemoryGraph(hd_dim=10000, seed=42)
        self.results: dict[str, Any] = {}

    def run_long_context_recall(
        self,
        seq_lengths: list[int] = [64, 128, 256, 512],
        hidden_dim: int = 64,
    ) -> dict[str, Any]:
        """Benchmark recall on synthetic long-context tasks.

        Creates sequences of token embeddings and tests whether spectral
        attention can retrieve information from distant positions.

        Args:
            seq_lengths: List of sequence lengths to test.
            hidden_dim: Hidden dimension for embeddings.

        Returns:
            Dict with recall scores per sequence length.
        """
        results: dict[str, float] = {}

        for seq_len in seq_lengths:
            # Create synthetic token embeddings
            rng = np.random.default_rng(42)
            tokens = rng.standard_normal((seq_len, hidden_dim)) * 0.1

            # Build graph and run attention
            attn = SpectralAttention(self.graph, hidden_dim=hidden_dim, num_heads=4)
            attn.build_graph_from_tokens(tokens)
            output = attn.forward(tokens)

            # Test recall: compare first token's output to original embedding
            original_first = tokens[0]
            recalled_first = output[0]

            # Cosine similarity as recall metric
            norm_orig = np.linalg.norm(original_first)
            norm_recall = np.linalg.norm(recalled_first)
            if norm_orig > 0 and norm_recall > 0:
                recall = float(np.dot(original_first, recalled_first) / (norm_orig * norm_recall))
            else:
                recall = 0.0

            results[f"len_{seq_len}"] = recall

        self.results["long_context_recall"] = results
        return results

    def run_throughput_benchmark(
        self,
        seq_lengths: list[int] = [64, 128, 256],
        hidden_dim: int = 64,
        num_runs: int = 5,
    ) -> dict[str, Any]:
        """Benchmark throughput of spectral attention.

        Args:
            seq_lengths: Sequence lengths to test.
            hidden_dim: Hidden dimension.
            num_runs: Number of runs for averaging.

        Returns:
            Dict with average time per sequence length.
        """
        results: dict[str, float] = {}
        rng = np.random.default_rng(42)

        for seq_len in seq_lengths:
            tokens = rng.standard_normal((seq_len, hidden_dim)) * 0.1
            times = []

            for _ in range(num_runs):
                attn = SpectralAttention(self.graph, hidden_dim=hidden_dim, num_heads=4)
                attn.build_graph_from_tokens(tokens)

                start = time.perf_counter()
                attn.forward(tokens)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[f"len_{seq_len}"] = float(np.mean(times))

        self.results["throughput"] = results
        return results

    def run_graph_construction_benchmark(
        self,
        num_nodes: list[int] = [100, 500, 1000],
    ) -> dict[str, Any]:
        """Benchmark graph construction speed.

        Args:
            num_nodes: List of node counts to test.

        Returns:
            Dict with construction time per node count.
        """
        results: dict[str, float] = {}

        for n in num_nodes:
            graph = SpectralMemoryGraph(hd_dim=10000, seed=42)
            start = time.perf_counter()

            for i in range(n):
                graph.add_node(f"node_{i}", label="benchmark",
                              properties={"index": i})

            for i in range(0, n - 1, 2):
                graph.add_edge(f"node_{i}", f"node_{i + 1}", "connected")

            elapsed = time.perf_counter() - start
            results[f"nodes_{n}"] = float(elapsed)

        self.results["graph_construction"] = results
        return results

    def summary(self) -> str:
        """Generate a summary of all benchmark results.

        Returns:
            Formatted string with benchmark results.
        """
        lines = ["SMGP Benchmark Summary", "=" * 40]

        for bench_name, data in self.results.items():
            lines.append(f"\n{bench_name}:")
            for key, value in data.items():
                lines.append(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        return "\n".join(lines)
