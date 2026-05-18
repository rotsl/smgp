"""Tests for the NeuroSymbolicPlanner module."""
import pytest

from smgp.core.graph import SpectralMemoryGraph
from smgp.reasoning.planner import NeuroSymbolicPlanner


@pytest.fixture
def planner_graph():
    g = SpectralMemoryGraph(hd_dim=100, seed=42)
    g.add_node("A", label="concept")
    g.add_node("B", label="concept")
    g.add_node("C", label="concept")
    g.add_edge("A", "B", "implies")
    return g


class TestNeuroSymbolicPlanner:
    def test_init(self, planner_graph):
        planner = NeuroSymbolicPlanner(planner_graph)
        assert planner.graph is planner_graph

    def test_register_rule(self, planner_graph):
        planner = NeuroSymbolicPlanner(planner_graph)
        rule = {
            "name": "test_rule",
            "lhs": {"nodes": [{"label": "concept"}], "edges": []},
            "rhs": {"nodes": [{"label": "concept"}], "edges": []},
            "interface": {"node_indices": [0]},
        }
        planner.register_rule(rule)
        assert len(planner._available_rules) == 1

    def test_plan_no_match(self, planner_graph):
        planner = NeuroSymbolicPlanner(planner_graph)
        # Goal that can't be satisfied
        plan = planner.plan("path from A to ZZZ", max_steps=2)
        assert isinstance(plan, list)

    def test_execute_plan_empty(self, planner_graph):
        planner = NeuroSymbolicPlanner(planner_graph)
        result = planner.execute_plan([])
        assert result is planner_graph

    def test_goal_satisfied(self, planner_graph):
        planner = NeuroSymbolicPlanner(planner_graph)
        # "path from A to B" should be satisfied (direct edge)
        assert planner._goal_satisfied("path from A to B") is True

    def test_goal_not_satisfied(self, planner_graph):
        planner = NeuroSymbolicPlanner(planner_graph)
        assert planner._goal_satisfied("path from A to NONEXISTENT") is False
