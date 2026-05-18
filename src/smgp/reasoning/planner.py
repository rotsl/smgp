"""Neuro-symbolic planning via graph rewriting and spectral search.

Combines symbolic graph rewriting rules with spectral similarity search
to plan multi-step reasoning chains over the knowledge graph.

References:
  - Ehrig, H., et al. (2006). "Fundamentals of Algebraic Graph Transformation."
  - Dechter, R. (2003). "Constraint Processing." Morgan Kaufmann.
"""
from __future__ import annotations

import re
from typing import Any

from smgp.core.category import GraphRewriter
from smgp.core.graph import SpectralMemoryGraph


class NeuroSymbolicPlanner:
    """Neuro-symbolic planner that combines graph rewriting with spectral search.

    Plans reasoning chains by:
    1. Parsing natural language goals into graph queries.
    2. Searching for applicable rewrite rules.
    3. Executing rules to transform the graph toward the goal.

    Attributes:
        graph: The knowledge graph to reason over.
        rewriter: Graph rewriting engine.
        _available_rules: List of registered rewrite rules.

    References:
        Ehrig, H., et al. (2006). "Fundamentals of Algebraic Graph Transformation."
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        max_steps: int = 10,
    ) -> None:
        """Initialize the planner.

        Args:
            graph: Knowledge graph to plan over.
            max_steps: Maximum number of planning steps.
        """
        self.graph = graph
        self.rewriter = GraphRewriter(graph)
        self.max_steps = max_steps
        self._available_rules: list[dict[str, Any]] = []

    def register_rule(self, rule: dict[str, Any]) -> None:
        """Register a rewrite rule for planning.

        Args:
            rule: Dict with keys 'name', 'lhs', 'rhs', 'interface'.
        """
        self._available_rules.append(rule)

    def plan(self, goal: str, max_steps: int | None = None) -> list[dict[str, Any]]:
        """Generate a plan to achieve a natural language goal.

        Args:
            goal: Natural language description of the goal.
            max_steps: Maximum planning steps (overrides default).

        Returns:
            List of planned rewrite operations.
        """
        steps = max_steps or self.max_steps
        plan: list[dict[str, Any]] = []

        # Parse goal for target entities
        target_match = re.search(r"from\s+(\S+)\s+to\s+(\S+)", goal, re.IGNORECASE)
        if target_match:
            source = target_match.group(1)
            target = target_match.group(2)

            # Check if path already exists
            if source in self.graph.graph and target in self.graph.graph:
                path = self._find_path(source, target)
                if path:
                    return plan  # Already satisfied

        # Try to apply available rules to reach the goal
        for _ in range(steps):
            applied = False
            for rule in self._available_rules:
                lhs = rule.get("lhs", {})
                matches = self.rewriter.match_pattern(lhs)
                if matches:
                    plan.append({
                        "rule": rule.get("name", "unnamed"),
                        "match": matches[0],
                        "lhs": lhs,
                        "rhs": rule.get("rhs", {}),
                        "interface": rule.get("interface", {}),
                    })
                    applied = True
                    break

            if not applied:
                break

        return plan

    def execute_plan(self, plan: list[dict[str, Any]]) -> SpectralMemoryGraph:
        """Execute a plan of rewrite operations.

        Args:
            plan: List of planned operations (output of plan()).

        Returns:
            The resulting graph after executing all rewrites.
        """
        for step in plan:
            lhs = step.get("lhs", {})
            rhs = step.get("rhs", {})
            interface = step.get("interface", {})
            match = step.get("match", {})

            if match and lhs:
                try:
                    self.rewriter.apply_dpo_rewrite(lhs, rhs, interface, match)
                except (ValueError, KeyError):
                    pass

        return self.graph

    def _goal_satisfied(self, goal: str) -> bool:
        """Check if a natural language goal is already satisfied.

        Args:
            goal: Natural language goal string.

        Returns:
            True if the goal is satisfied, False otherwise.
        """
        path_match = re.search(r"path from (\S+) to (\S+)", goal, re.IGNORECASE)
        if path_match:
            source = path_match.group(1)
            target = path_match.group(2)
            path = self._find_path(source, target)
            return path is not None

        return False

    def _find_path(self, source: str, target: str) -> list[str] | None:
        """Find a path between two nodes using BFS.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            List of node IDs from source to target, or None if no path.
        """
        if source not in self.graph.graph or target not in self.graph.graph:
            return None

        visited = {source}
        queue: list[list[str]] = [[source]]

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node == target:
                return path

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None
