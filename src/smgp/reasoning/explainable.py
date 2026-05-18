"""Explainable reasoning with proof sub-graphs.

Extends the base :class:`~smgp.reasoning.verifier.ClaimVerifier` and
:class:`~smgp.reasoning.planner.NeuroSymbolicPlanner` so that every
result includes a ``"proof_subgraph"`` key containing the edges that
form the justification.

The proof sub-graph enables human inspection and audit-trail logging.
"""
from __future__ import annotations

from typing import Any

from smgp.reasoning.planner import NeuroSymbolicPlanner
from smgp.reasoning.verifier import ClaimVerifier


class ExplainableClaimVerifier(ClaimVerifier):
    """Claim verifier that returns proof sub-graphs.

    The ``verify`` result includes a ``"proof_subgraph"`` key — a list
    of edge dicts that form the proof path.

    Example
    -------
    ::
        verifier = ExplainableClaimVerifier(graph)
        result = verifier.verify("Socrates taught Plato")
        # result["proof_subgraph"] == [
        #     {"source": "Socrates", "target": "Plato", "relation": "taught"}
        # ]
    """

    def verify(self, claim: str) -> dict[str, Any]:
        """Verify *claim* and attach a proof sub-graph.

        Parameters
        ----------
        claim : str
            Natural-language or triple-format claim.

        Returns
        -------
        dict
            Same keys as the parent :meth:`verify`, plus:
              - ``"proof_subgraph"`` — list of edge dicts.
        """
        # Run parent verification logic
        base_result = super().verify(claim)

        # Build proof sub-graph by extracting edges along the proof path
        proof_edges = self._extract_proof_edges(claim)

        base_result["proof_subgraph"] = proof_edges
        return base_result

    def _extract_proof_edges(self, claim: str) -> list[dict[str, Any]]:
        """Try to extract the edge(s) that support *claim*.

        Strategy:
          1. Parse claim as triple "Subject relation Object".
          2. If both subject and object are in the graph, enumerate the
             matching edges.
          3. If no direct edge, try to find a path and collect edges.

        Parameters
        ----------
        claim : str

        Returns
        -------
        list[dict]
            Each dict has keys ``"source"``, ``"target"``, ``"relation"``.
        """
        edges: list[dict[str, Any]] = []

        # Try triple format: "Subject relation Object"
        parts = claim.strip().split(None, 2)
        if len(parts) == 3:
            subject, relation, obj = parts
            if (subject in self.graph.graph and obj in self.graph.graph):
                for src, tgt, key, edata in self.graph.graph.edges(
                    subject, data=True, keys=True
                ):
                    rel = edata.get("relation", "")
                    if tgt == obj and (relation == "" or rel == relation):
                        edges.append({
                            "source": src,
                            "target": tgt,
                            "relation": rel,
                        })

        # Try natural language parsing (same patterns as parent)
        if not edges:
            parsed = self._parse_claim(claim)
            if parsed:
                subject = parsed["subject"]
                obj = parsed.get("object", "")
                if subject in self.graph.graph and obj in self.graph.graph:
                    for src, tgt, key, edata in self.graph.graph.edges(
                        subject, data=True, keys=True
                    ):
                        if tgt == obj:
                            edges.append({
                                "source": src,
                                "target": tgt,
                                "relation": edata.get("relation", ""),
                            })

        # Try multi-hop path
        if not edges and len(parts) >= 2:
            path = self.find_path(parts[0], parts[-1])
            if path and len(path) > 1:
                for i in range(len(path) - 1):
                    s, t = path[i], path[i + 1]
                    rel = ""
                    for _, tgt, key, edata in self.graph.graph.edges(
                        s, data=True, keys=True
                    ):
                        if tgt == t:
                            rel = edata.get("relation", "")
                            break
                    edges.append({"source": s, "target": t, "relation": rel})

        return edges


class ExplainableNeuroSymbolicPlanner(NeuroSymbolicPlanner):
    """Neuro-symbolic planner that annotates each step with a proof sub-graph.

    Each step dict returned by :meth:`plan` includes a
    ``"proof_subgraph"`` key listing the edges relevant to that step.
    """

    def plan(
        self,
        goal: str,
        max_steps: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a plan and annotate each step with proof edges.

        Parameters
        ----------
        goal : str
            Natural language goal.
        max_steps : int or None
            Maximum planning steps.

        Returns
        -------
        list[dict]
            Planned steps, each augmented with ``"proof_subgraph"``.
        """
        base_plan = super().plan(goal, max_steps=max_steps)

        # Annotate each step with the proof sub-graph
        annotated: list[dict[str, Any]] = []
        for step in base_plan:
            proof_edges = self._step_proof_edges(step)
            step_copy = dict(step)
            step_copy["proof_subgraph"] = proof_edges
            annotated.append(step_copy)

        return annotated

    def _step_proof_edges(self, step: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract edges relevant to a single planning step.

        Looks at the ``"match"`` information in *step* to find
        the edges involved in the pattern match.

        Parameters
        ----------
        step : dict
            A single step from the parent's plan output.

        Returns
        -------
        list[dict]
            Edge dicts with ``"source"``, ``"target"``, ``"relation"``.
        """
        edges: list[dict[str, Any]] = []
        match = step.get("match", {})

        if not match:
            return edges

        # The match dict typically contains node IDs keyed by variable names
        # Try to find edges between matched nodes
        matched_nodes = list(match.values())
        for i, src in enumerate(matched_nodes):
            if src not in self.graph.graph:
                continue
            for tgt in matched_nodes[i + 1:]:
                if tgt not in self.graph.graph:
                    continue
                for _, t, key, edata in self.graph.graph.edges(
                    src, data=True, keys=True
                ):
                    if t == tgt:
                        edges.append({
                            "source": src,
                            "target": t,
                            "relation": edata.get("relation", ""),
                        })

        return edges
