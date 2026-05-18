"""ASV benchmarks for SpectralMemoryGraph operations.

Benchmarks graph construction, similarity search, and memory usage
for the core knowledge graph data structure.

References:
    Angles, R. & Gutierrez, C. (2008). "Survey of Graph Database Models."
        ACM Computing Surveys, 40(1), 1-39.
"""
from __future__ import annotations

import numpy as np

from smgp.core.graph import SpectralMemoryGraph


class GraphBenchmarks:
    """Benchmarks for graph construction, querying, and memory."""

    params = [100, 500, 1000, 5000]
    param_names = ["n"]

    def setup(self, n):
        """Create a fresh graph instance for each benchmark run."""
        self.graph = SpectralMemoryGraph(hd_dim=10000, seed=42)

    def teardown(self, n):
        """Destroy graph to free memory between runs."""
        del self.graph

    def time_graph_construction(self, n):
        """Benchmark adding n nodes and n-1 edges.

        Measures the time to construct a graph with n nodes connected
        in a chain topology (n-1 sequential edges).
        """
        graph = self.graph
        for i in range(n):
            graph.add_node(f"node_{i}", label="benchmark",
                           properties={"index": i})
        for i in range(n - 1):
            graph.add_edge(f"node_{i}", f"node_{i + 1}", "connected")

    def time_similarity_search(self, n):
        """Benchmark querying similar nodes in an n-node graph.

        Constructs the graph, generates a query vector, and performs
        a k-nearest-neighbour similarity search over all nodes.
        """
        graph = self.graph
        for i in range(n):
            graph.add_node(f"node_{i}", label="benchmark",
                           properties={"index": i})
        for i in range(n - 1):
            graph.add_edge(f"node_{i}", f"node_{i + 1}", "connected")
        # Query with a random HD vector
        query_vec = graph.hd.generate(1)[0]
        graph.query_similar(query_vec, k=10)

    def peakmem_graph_construction(self, n):
        """Measure peak memory during graph construction with n nodes."""
        graph = self.graph
        for i in range(n):
            graph.add_node(f"node_{i}", label="benchmark",
                           properties={"index": i})
        for i in range(n - 1):
            graph.add_edge(f"node_{i}", f"node_{i + 1}", "connected")
