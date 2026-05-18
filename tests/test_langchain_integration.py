"""Tests for the LangChain integration module."""
from smgp.integration.langchain import SMGPMemory, SMGPVerifierTool


class TestSMGPMemory:
    def test_save_context(self):
        memory = SMGPMemory(hd_dim=100)
        memory.save_context(
            {"input": "Hello"},
            {"output": "Hi there!"},
        )
        assert memory.turn_count == 1

    def test_load_memory_variables(self):
        memory = SMGPMemory(hd_dim=100)
        memory.save_context(
            {"input": "What is AI?"},
            {"output": "AI is artificial intelligence."},
        )
        memory.save_context(
            {"input": "Tell me more"},
            {"output": "AI involves machine learning."},
        )
        result = memory.load_memory_variables({"input": "What is AI?"})
        assert "history" in result
        assert len(result["history"]) > 0

    def test_clear(self):
        memory = SMGPMemory(hd_dim=100)
        memory.save_context({"input": "test"}, {"output": "response"})
        assert memory.turn_count == 1
        memory.clear()
        assert memory.turn_count == 0
        assert memory.graph.num_nodes == 0

    def test_memory_variables(self):
        memory = SMGPMemory(hd_dim=100)
        assert memory.memory_variables == ["history"]


class TestSMGPVerifierTool:
    def test_verify_claim(self):
        tool = SMGPVerifierTool()
        tool.add_knowledge("Paris", "capital_of", "France")
        result = tool._run("Paris capital_of France")
        assert "VERIFIED" in result

    def test_verify_unverified(self):
        tool = SMGPVerifierTool()
        result = tool._run("Tokyo is the capital of Mars")
        assert "UNVERIFIED" in result

    def test_add_knowledge(self):
        tool = SMGPVerifierTool()
        tool.add_knowledge("Alice", "knows", "Bob")
        assert tool.graph.num_nodes == 2
        assert tool.graph.num_edges == 1
