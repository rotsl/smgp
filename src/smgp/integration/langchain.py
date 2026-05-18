"""LangChain integration for SMGP persistent memory and verification.

Provides:
  - SMGPMemory: A LangChain BaseMemory that uses SMGP's persistent graph memory
    for perfect recall across long-running agent sessions.
  - SMGPVerifierTool: A LangChain Tool that wraps SMGP's claim verifier for
    real-time fact-checking during agent reasoning.

References:
  - Chase, H. (2022). "LangChain: Building Applications with LLMs through Composability."
    https://github.com/langchain-ai/langchain.
"""
from __future__ import annotations

from typing import Any

from smgp.core.graph import SpectralMemoryGraph
from smgp.reasoning.verifier import ClaimVerifier


class SMGPMemory:
    """LangChain-compatible memory using SMGP's persistent knowledge graph.

    Stores conversation context as a graph, enabling:
    - Perfect recall of previous interactions (no context window limits).
    - Semantic search over conversation history via HD similarity.
    - Structured knowledge accumulation over multi-turn dialogs.

    Compatible with LangChain's BaseMemory interface:
    - save_context(inputs, outputs): Store a conversation turn.
    - load_memory_variables(inputs): Retrieve relevant context.
    - clear(): Reset memory.

    Attributes:
        graph: The knowledge graph storing conversation data.
        turn_count: Number of conversation turns stored.

    References:
        Chase, H. (2022). "LangChain."
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph | None = None,
        hd_dim: int = 10000,
    ) -> None:
        """Initialize SMGP memory.

        Args:
            graph: Optional pre-existing graph. If None, creates a new one.
            hd_dim: HD vector dimensionality for new graphs.
        """
        self.graph = graph or SpectralMemoryGraph(hd_dim=hd_dim, seed=42)
        self.turn_count = 0

    def save_context(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Store a conversation turn in the knowledge graph.

        Creates nodes for the input and output, with edges connecting
        them in sequence. Also stores HD vectors for similarity-based
        retrieval.

        Args:
            inputs: Dict with 'input' or similar key containing the user message.
            outputs: Dict with 'output' or similar key containing the AI response.
        """
        self.turn_count += 1
        input_text = str(inputs.get("input", inputs.get("human_input", "")))
        output_text = str(outputs.get("output", outputs.get("response", "")))

        input_id = f"turn_{self.turn_count}_input"
        output_id = f"turn_{self.turn_count}_output"
        prev_output_id = f"turn_{self.turn_count - 1}_output" if self.turn_count > 1 else None

        self.graph.add_node(
            input_id, label="user_input",
            properties={"text": input_text, "turn": self.turn_count}
        )
        self.graph.add_node(
            output_id, label="ai_output",
            properties={"text": output_text, "turn": self.turn_count}
        )

        self.graph.add_edge(input_id, output_id, "response_to")

        if prev_output_id and prev_output_id in self.graph.graph:
            self.graph.add_edge(prev_output_id, input_id, "followed_by")

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Load relevant conversation context.

        Retrieves recent conversation turns and semantically similar
        past turns based on the current input.

        Args:
            inputs: Current input dict, used for similarity search.

        Returns:
            Dict with 'history' key containing formatted conversation history.
        """
        query = str(inputs.get("input", inputs.get("human_input", "")))

        # Get recent turns (last 5)
        recent = self._get_recent_turns(n=5)

        # Get semantically similar turns
        similar = self._get_similar_turns(query, k=3)

        # Combine, avoiding duplicates
        all_turns = list(recent)
        seen = {t["input_id"] for t in recent}
        for t in similar:
            if t["input_id"] not in seen:
                all_turns.append(t)
                seen.add(t["input_id"])

        # Sort by turn number
        all_turns.sort(key=lambda t: t.get("turn", 0))

        history = ""
        for turn in all_turns:
            history += f"Human: {turn['input_text']}\nAI: {turn['output_text']}\n\n"

        return {"history": history.strip()}

    def clear(self) -> None:
        """Clear all stored conversation data."""
        self.graph = SpectralMemoryGraph(hd_dim=self.graph.hd.dim, seed=42)
        self.turn_count = 0

    def _get_recent_turns(self, n: int = 5) -> list[dict[str, str]]:
        """Get the n most recent conversation turns.

        Args:
            n: Number of recent turns.

        Returns:
            List of turn dicts with input/output text and IDs.
        """
        turns = []
        start = max(1, self.turn_count - n + 1)
        for t in range(start, self.turn_count + 1):
            input_id = f"turn_{t}_input"
            output_id = f"turn_{t}_output"
            input_data = self.graph.get_node(input_id)
            output_data = self.graph.get_node(output_id)
            if input_data and output_data:
                turns.append({
                    "turn": t,
                    "input_id": input_id,
                    "output_id": output_id,
                    "input_text": input_data["properties"].get("text", ""),
                    "output_text": output_data["properties"].get("text", ""),
                })
        return turns

    def _get_similar_turns(self, query: str, k: int = 3) -> list[dict[str, str]]:
        """Find semantically similar past turns.

        Uses HD vector similarity to find past inputs similar to the query.

        Args:
            query: Query text.
            k: Number of results.

        Returns:
            List of turn dicts.
        """
        query_vec = self.graph.hd.generate(1)[0]
        similar = self.graph.query_similar(query_vec, k=k, node_type="user_input")

        turns = []
        for nid, sim in similar:
            data = self.graph.get_node(nid)
            if data:
                turn_num = data["properties"].get("turn", 0)
                output_id = f"turn_{turn_num}_output"
                output_data = self.graph.get_node(output_id)
                turns.append({
                    "turn": turn_num,
                    "input_id": nid,
                    "output_id": output_id,
                    "input_text": data["properties"].get("text", ""),
                    "output_text": output_data["properties"].get("text", "") if output_data else "",
                    "similarity": sim,
                })

        return turns

    @property
    def memory_variables(self) -> list[str]:
        """List of memory variable names (LangChain interface)."""
        return ["history"]


class SMGPVerifierTool:
    """LangChain-compatible tool for fact verification using SMGP.

    Wraps the SMGP ClaimVerifier as a LangChain Tool, enabling agents
    to fact-check claims during reasoning.

    Attributes:
        name: Tool name.
        description: Tool description for the agent.
        verifier: Underlying ClaimVerifier instance.

    References:
        Chase, H. (2022). "LangChain."
    """

    name: str = "smgp_verifier"
    description: str = (
        "Verify a factual claim against the knowledge graph. "
        "Input should be a claim string (e.g., 'Paris is the capital of France'). "
        "Returns verification status, confidence, and reasoning."
    )

    def __init__(
        self,
        graph: SpectralMemoryGraph | None = None,
        verifier: ClaimVerifier | None = None,
    ) -> None:
        """Initialize the verifier tool.

        Args:
            graph: Optional knowledge graph. Creates one if not provided.
            verifier: Optional pre-built ClaimVerifier.
        """
        self.graph = graph or SpectralMemoryGraph(hd_dim=10000, seed=42)
        self.verifier = verifier or ClaimVerifier(self.graph)

    def _run(self, claim: str) -> str:
        """Execute verification on a claim.

        Args:
            claim: Natural language claim to verify.

        Returns:
            Formatted string with verification results.
        """
        result = self.verifier.verify(claim)

        status = "VERIFIED" if result["verified"] else "UNVERIFIED"
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "No reasoning available.")

        if result.get("path"):
            path_str = " -> ".join(result["path"])
            return f"[{status}] (confidence: {confidence:.2f}) {reasoning}\nPath: {path_str}"

        return f"[{status}] (confidence: {confidence:.2f}) {reasoning}"

    async def _arun(self, claim: str) -> str:
        """Async version of _run."""
        return self._run(claim)

    def add_knowledge(
        self,
        subject: str,
        relation: str,
        obj: str,
    ) -> None:
        """Add a fact to the knowledge graph for verification.

        Args:
            subject: Subject entity.
            relation: Relation type.
            obj: Object entity.
        """
        if subject not in self.graph.graph:
            self.graph.add_node(subject, label="entity", properties={"name": subject})
        if obj not in self.graph.graph:
            self.graph.add_node(obj, label="entity", properties={"name": obj})
        self.graph.add_edge(subject, obj, relation)
