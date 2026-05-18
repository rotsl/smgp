"""End-to-end integration test for the full SMGP pipeline."""
import numpy as np

from smgp.attention.spectral_attn import SpectralAttention
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.hyperdim import HyperdimensionalMemory
from smgp.core.spectral import SpectralMethods
from smgp.integration.huggingface import SMGPConfig, SMGPForCausalLM
from smgp.integration.langchain import SMGPMemory
from smgp.memory.associative import AssociativeMemory
from smgp.memory.lifecycle import MemoryLifecycle
from smgp.reasoning.verifier import ClaimVerifier


class TestEndToEnd:
    def test_full_pipeline(self):
        """Test the complete SMGP pipeline from knowledge to verification."""
        # Step 1: Build a knowledge graph
        graph = SpectralMemoryGraph(hd_dim=500, seed=42)
        graph.add_node("Python", label="language", properties={"name": "Python"})
        graph.add_node("programming", label="concept")
        graph.add_node("AI", label="concept")
        graph.add_node("machine_learning", label="concept")
        graph.add_edge("Python", "programming", "is_a")
        graph.add_edge("Python", "AI", "used_in")
        graph.add_edge("AI", "machine_learning", "includes")

        assert graph.num_nodes == 4
        assert graph.num_edges == 3

        # Step 2: Run spectral analysis
        spectral = SpectralMethods(graph, num_eigenvalues=3)
        L = spectral.compute_laplacian()
        assert L.shape == (4, 4)

        eigenvalues, eigenvectors = spectral.compute_eigen()
        assert len(eigenvalues) == 3

        # Step 3: Run spectral attention
        tokens = np.random.randn(4, 32) * 0.1
        attn = SpectralAttention(graph, hidden_dim=32, num_heads=2)
        attn.build_graph_from_tokens(tokens)
        output = attn.forward(tokens)
        assert output.shape == (4, 32)

        # Step 4: Verify claims
        verifier = ClaimVerifier(graph)
        result = verifier.verify("Python is_a programming")
        assert "verified" in result
        assert "confidence" in result

        # Step 5: Find paths
        path = verifier.find_path("Python", "machine_learning")
        assert path is not None
        assert "Python" in path
        assert "machine_learning" in path

    def test_associative_memory_pipeline(self):
        """Test associative memory store and recall."""
        hd = HyperdimensionalMemory(dim=500, seed=42)
        assoc = AssociativeMemory(hd=hd)
        assoc.store("capital_france", "Paris")
        assoc.store("capital_uk", "London")
        assoc.store("capital_japan", "Tokyo")

        results = assoc.recall("capital_france", k=1)
        assert len(results) >= 1
        assert results[0][0] == "capital_france"

    def test_huggingface_pipeline(self):
        """Test HuggingFace model forward and generation."""
        config = SMGPConfig(
            hd_dim=100,
            hidden_dim=32,
            num_heads=2,
            vocab_size=50,
            max_position_embeddings=16,
        )
        model = SMGPForCausalLM(config)
        input_ids = np.array([[1, 2, 3, 4, 5]])
        output = model.forward(input_ids)
        assert output["logits"].shape == (1, 5, 50)

        generated = model.generate(input_ids[0], max_length=3)
        assert generated.shape[0] == 8

    def test_langchain_memory_pipeline(self):
        """Test LangChain memory across sessions."""
        memory = SMGPMemory(hd_dim=100)
        memory.save_context({"input": "Hello"}, {"output": "Hi!"})
        memory.save_context({"input": "How are you?"}, {"output": "I'm good!"})

        result = memory.load_memory_variables({"input": "Hello"})
        assert "history" in result
        assert "Hello" in result["history"]

        # Clear and verify
        memory.clear()
        assert memory.turn_count == 0

    def test_memory_lifecycle(self):
        """Test memory pruning and lifecycle."""
        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        # Add connected cluster
        for i in range(5):
            graph.add_node(f"node_{i}", label="cluster")
            if i > 0:
                graph.add_edge(f"node_{i-1}", f"node_{i}", "connected")

        lifecycle = MemoryLifecycle(graph)
        eval_result = lifecycle.evaluate()
        assert eval_result["num_nodes"] == 5

        pruned = lifecycle.prune(strategy="isolated", threshold=0.1)
        assert pruned.num_nodes == 5  # All connected, none isolated

    def test_graph_io_roundtrip(self, tmp_path):
        """Test saving and loading a graph."""
        from smgp.utils.io import load_graph, save_graph

        graph = SpectralMemoryGraph(hd_dim=100, seed=42)
        graph.add_node("X", label="test", properties={"val": 42})
        graph.add_node("Y", label="test")
        graph.add_edge("X", "Y", "connects")

        save_path = str(tmp_path / "test_graph")
        saved = save_graph(graph, save_path, format="json")

        loaded = load_graph(saved, format="json")
        assert loaded.num_nodes == 2
        assert loaded.num_edges == 1
        assert loaded.get_node("X")["properties"]["val"] == 42
