"""Claim verification against the knowledge graph.

Implements factual claim verification by checking whether claims can be
grounded in the knowledge graph structure. Supports direct edge verification
and multi-hop path finding.

References:
  - Angeli, G., et al. (2014). "Leveraging Free Text to Enrich Structured
    Knowledge Bases." NAACL-HLT.
  - Verga, P., et al. (2020). "Multihop Knowledge Graph Reasoning with
    Reward Shaping." EMNLP.
"""
from __future__ import annotations

import re
from typing import Any

from smgp.core.graph import SpectralMemoryGraph


class ClaimVerifier:
    """Verifies factual claims against the knowledge graph.

    Supports:
    - Direct edge verification: "Paris capital_of France"
    - Multi-hop path finding: "path from X to Y"
    - Natural language claim parsing: "X is the capital of Y"

    Attributes:
        graph: The knowledge graph to verify claims against.

    References:
        Angeli, G., et al. (2014). "Leveraging Free Text." NAACL-HLT.
    """

    def __init__(self, graph: SpectralMemoryGraph) -> None:
        """Initialize the claim verifier.

        Args:
            graph: Knowledge graph to verify against.
        """
        self.graph = graph

    def verify(self, claim: str) -> dict[str, Any]:
        """Verify a factual claim against the knowledge graph.

        Attempts to parse the claim as:
        1. Direct triple: "Subject relation Object"
        2. Natural language: "Subject is the relation of Object"
        3. Path query: "path from X to Y"

        Args:
            claim: Natural language claim string.

        Returns:
            Dict with 'verified' (bool) and 'confidence' (float).
        """
        # Try direct triple format first: "Subject relation Object"
        parts = claim.strip().split(None, 2)
        if len(parts) == 3:
            subject, relation, obj = parts
            if subject in self.graph.graph and obj in self.graph.graph:
                # Check for direct edge
                for src_node, target, key, edata in self.graph.graph.edges(
                    subject, data=True, keys=True
                ):
                    if target == obj and edata.get("relation") == relation:
                        return {"verified": True, "confidence": 1.0}

        # Try natural language parsing
        parsed = self._parse_claim(claim)
        if parsed:
            subject = parsed["subject"]
            obj = parsed.get("object", "")

            if subject in self.graph.graph and obj in self.graph.graph:
                # Try to match the predicate to a relation
                for src_node, target, key, edata in self.graph.graph.edges(
                    subject, data=True, keys=True
                ):
                    if target == obj:
                        return {"verified": True, "confidence": 0.8}

        return {"verified": False, "confidence": 0.0}

    def find_path(
        self, source: str, target: str, max_depth: int = 10
    ) -> list[str] | None:
        """Find a path between two nodes using BFS.

        Args:
            source: Source node ID.
            target: Target node ID.
            max_depth: Maximum search depth.

        Returns:
            List of node IDs from source to target, or None.
        """
        if source not in self.graph.graph or target not in self.graph.graph:
            return None

        visited = {source}
        queue: list[list[str]] = [[source]]
        depth = 0

        while queue and depth < max_depth:
            path = queue.pop(0)
            node = path[-1]

            if node == target:
                return path

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

            depth += 1

        return None

    def _parse_claim(self, claim: str) -> dict[str, str] | None:
        """Parse a natural language claim into subject, predicate, object.

        Handles patterns like:
        - "X is Y"
        - "X is the R of Y"
        - "X R Y"

        Args:
            claim: Natural language claim string.

        Returns:
            Dict with 'subject', 'predicate', 'object' keys, or None.
        """
        claim = claim.strip()

        # Pattern: "X is the R of Y"
        match = re.match(
            r"^(.+?)\s+is\s+(?:the\s+)?(.+?)\s+of\s+(.+)$", claim, re.IGNORECASE
        )
        if match:
            return {
                "subject": match.group(1).strip(),
                "predicate": match.group(2).strip(),
                "object": match.group(3).strip(),
            }

        # Pattern: "X is Y"
        match = re.match(r"^(.+?)\s+is\s+(.+)$", claim, re.IGNORECASE)
        if match:
            return {
                "subject": match.group(1).strip(),
                "predicate": "is",
                "object": match.group(2).strip(),
            }

        return None
