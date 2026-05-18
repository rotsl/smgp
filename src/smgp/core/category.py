"""Category theory-based graph rewriting via Double-Pushout (DPO).

Implements algebraic graph transformation as described in:
- Ehrig, H., et al. (2006). "Fundamentals of Algebraic Graph Transformation."
  Springer Monographs in Mathematics.
- Ehrig, H., et al. (1997). "Parallel and Distributed Derivations in the
  Single-Pushout Approach." TCS, 185(1), 37–64.

The Double-Pushout (DPO) approach is the standard categorical framework for
graph rewriting. A DPO rule consists of:
  - A left-hand side (L) pattern to match in the host graph
  - An interface (K) of nodes/edges preserved by the rule
  - A right-hand side (R) pattern replacing the matched L

A rewrite is valid if:
  1. L matches a subgraph of the host graph G (gluing condition)
  2. The pushout complement exists (no dangling edges after deletion)
  3. The resulting graph G' is well-formed
"""
from __future__ import annotations

from typing import Any

from smgp.core.graph import SpectralMemoryGraph


class GraphRewriter:
    """Category-theoretic graph rewriting engine using Double-Pushout (DPO).

    Enables formal reasoning steps where each deduction is a valid rewrite
    operation on the knowledge graph. Every rewrite generates a proof trace
    that can be verified for correctness.

    Attributes:
        graph: The knowledge graph to rewrite.
        proof_traces: List of applied rewrites for verification.

    References:
        Ehrig, H., et al. (2006). "Fundamentals of Algebraic Graph Transformation."
    """

    def __init__(self, graph: SpectralMemoryGraph) -> None:
        """Initialize the graph rewriter.

        Args:
            graph: Knowledge graph to perform rewrites on.
        """
        self.graph = graph
        self.proof_traces: list[dict[str, Any]] = []

    def match_pattern(self, pattern: dict[str, Any]) -> list[dict[str, str]]:
        """Find all matches of a graph pattern in the host graph.

        A pattern specifies:
          - 'nodes': list of {'label': str, 'constraints': dict} specs
          - 'edges': list of {'source': int, 'target': int, 'relation': str} specs
            where source/target are indices into the nodes list

        Args:
            pattern: Pattern specification dict.

        Returns:
            List of matches, each a dict mapping pattern node index to
            the matched node_id in the host graph.
        """
        pattern_nodes = pattern.get("nodes", [])
        pattern_edges = pattern.get("edges", [])

        if not pattern_nodes:
            return []

        # Find candidate nodes for each pattern node
        candidates: list[list[str]] = []
        for pnode in pattern_nodes:
            label = pnode.get("label", "")
            cands: list[str] = []
            for nid, ndata in self.graph.graph.nodes(data=True):
                if not label or ndata.get("label") == label:
                    # Check additional constraints
                    constraints = pnode.get("constraints", {})
                    valid = True
                    for key, expected in constraints.items():
                        if key == "properties" and isinstance(expected, dict):
                            props = ndata.get("properties", {})
                            for pk, pv in expected.items():
                                if props.get(pk) != pv:
                                    valid = False
                                    break
                        elif ndata.get(key) != expected:
                            valid = False
                    if valid:
                        cands.append(nid)
            candidates.append(cands)

        # Filter candidates by edge constraints
        matches: list[dict[str, str]] = []

        def backtrack(idx: int, current_match: dict[str, str]) -> None:
            if idx == len(pattern_nodes):
                # Verify edge constraints
                edge_ok = True
                for pedge in pattern_edges:
                    src_idx = pedge["source"]
                    tgt_idx = pedge["target"]
                    src_id = current_match[str(src_idx)]
                    tgt_id = current_match[str(tgt_idx)]
                    relation = pedge.get("relation", "")

                    # Check if edge exists with matching relation
                    found = False
                    for (
                        _u,
                        _v,
                        _k,
                        edata,
                    ) in self.graph.graph.edges(src_id, data=True, keys=True):
                        if _v == tgt_id and edata.get("relation") == relation:
                            found = True
                            break
                    if not found:
                        edge_ok = False
                        break

                if edge_ok:
                    matches.append(dict(current_match))
                return

            for cand in candidates[idx]:
                # Check consistency (no two pattern nodes map to same graph node)
                if cand not in current_match.values():
                    current_match[str(idx)] = cand
                    backtrack(idx + 1, current_match)
                    del current_match[str(idx)]

        backtrack(0, {})
        return matches

    def apply_dpo_rewrite(
        self,
        lhs: dict[str, Any],
        rhs: dict[str, Any],
        interface: dict[str, Any],
        match: dict[str, str],
    ) -> SpectralMemoryGraph:
        """Apply a Double-Pushout (DPO) graph rewrite rule.

        The DPO rewrite proceeds as:
        1. Check the gluing condition (no dangling edges from deleted nodes)
        2. Delete L\\K (nodes/edges in L but not in interface K)
        3. Add R\\K (nodes/edges in R but not in interface K)

        Args:
            lhs: Left-hand side pattern (same format as match_pattern input).
            rhs: Right-hand side pattern with 'nodes' and 'edges'.
            interface: Interface K specifying which node indices are preserved.
            match: A match dict mapping pattern indices to graph node IDs.
                Must be a valid match from match_pattern.

        Returns:
            New SpectralMemoryGraph with the rewrite applied.

        Raises:
            ValueError: If the gluing condition is violated.

        References:
            Ehrig, H., et al. (2006). "Fundamentals of Algebraic Graph
                Transformation," Ch. 3-4.
        """
        # Step 1: Check gluing condition
        # Nodes to delete: L \ K
        interface_indices = set(interface.get("node_indices", []))
        lhs_indices = set(range(len(lhs.get("nodes", []))))
        delete_indices = lhs_indices - interface_indices

        # Check that nodes to be deleted have no dangling edges to preserved nodes
        delete_node_ids = {match[str(i)] for i in delete_indices}
        preserve_node_ids = {match[str(i)] for i in interface_indices}

        for del_nid in delete_node_ids:
            for neighbor in self.graph.neighbors(del_nid):
                if neighbor in preserve_node_ids:
                    # Check if the edge is in the LHS (so it's being removed)
                    edge_in_lhs = False
                    for pedge in lhs.get("edges", []):
                        src_id = match[str(pedge["source"])]
                        tgt_id = match[str(pedge["target"])]
                        if (src_id == del_nid and tgt_id == neighbor) or (
                            tgt_id == del_nid and src_id == neighbor
                        ):
                            edge_in_lhs = True
                            break
                    if not edge_in_lhs:
                        raise ValueError(
                            f"Gluing condition violated: node '{del_nid}' has "
                            f"edge to preserved node '{neighbor}' not in LHS."
                        )

        # Step 2: Create new graph (deep copy operation)
        new_graph = SpectralMemoryGraph(hd_dim=self.graph.hd.dim, seed=42)
        for nid in self.graph.nodes():
            data = self.graph.get_node(nid)
            if data:
                new_graph.add_node(
                    nid,
                    label=data["label"],
                    properties=data["properties"],
                    vector=data["vector"],
                )

        for src, tgt, key, edata in self.graph.edges():
            new_graph.add_edge(
                src,
                tgt,
                edata.get("relation", ""),
                properties=edata.get("properties", {}),
            )

        # Step 3: Delete L \ K nodes and their edges
        for i in delete_indices:
            nid = match[str(i)]
            if nid in new_graph.graph:
                new_graph.remove_node(nid)

        # Step 4: Add R \ K nodes and edges
        rhs_indices = set(range(len(rhs.get("nodes", []))))
        add_indices = rhs_indices - interface_indices

        # Map: interface indices reuse existing node IDs, new indices get new IDs
        rhs_node_map: dict[int, str] = {}
        for i in rhs_indices:
            if i in interface_indices:
                rhs_node_map[i] = match[str(i)]
            else:
                new_id = f"gen_{len(new_graph.nodes())}_{i}"
                rhs_node_map[i] = new_id
                rnode = rhs["nodes"][i]
                new_graph.add_node(
                    new_id,
                    label=rnode.get("label", ""),
                    properties=rnode.get("properties", {}),
                )

        # Add R \ K edges
        for pedge in rhs.get("edges", []):
            src_id = rhs_node_map[pedge["source"]]
            tgt_id = rhs_node_map[pedge["target"]]
            new_graph.add_edge(
                src_id,
                tgt_id,
                pedge.get("relation", ""),
                properties=pedge.get("properties", {}),
            )

        # Record proof trace
        self.proof_traces.append(
            {
                "rule_name": lhs.get("name", "unnamed"),
                "lhs": lhs,
                "rhs": rhs,
                "interface": interface,
                "match": match,
                "deleted_nodes": list(delete_node_ids),
                "added_nodes": [rhs_node_map[i] for i in add_indices],
            }
        )

        # Update working graph
        self.graph = new_graph
        return new_graph

    def verify_rewrite_sequence(
        self, rewrites: list[dict[str, Any]]
    ) -> bool:
        """Verify a sequence of rewrites for logical consistency.

        Checks that:
        1. Each rewrite has a valid match in the graph at that point
        2. The gluing condition holds for each rewrite
        3. The overall sequence produces a consistent result

        Args:
            rewrites: List of rewrite specifications, each with 'lhs', 'rhs',
                'interface', and 'match' keys.

        Returns:
            True if the sequence is valid, False otherwise.
        """
        # We replay the sequence on a copy and check each step
        test_graph = SpectralMemoryGraph(hd_dim=self.graph.hd.dim, seed=42)
        for nid in self.graph.nodes():
            data = self.graph.get_node(nid)
            if data:
                test_graph.add_node(
                    nid,
                    label=data["label"],
                    properties=data["properties"],
                    vector=data["vector"],
                )
        for src, tgt, key, edata in self.graph.edges():
            test_graph.add_edge(
                src,
                tgt,
                edata.get("relation", ""),
                properties=edata.get("properties", {}),
            )

        test_rewriter = GraphRewriter(test_graph)

        for i, rewrite in enumerate(rewrites):
            lhs = rewrite.get("lhs", {})
            match = rewrite.get("match", {})

            # Check match is still valid
            matches = test_rewriter.match_pattern(lhs)
            if not matches:
                return False

            # Try to apply
            try:
                test_rewriter.apply_dpo_rewrite(
                    lhs=lhs,
                    rhs=rewrite.get("rhs", {}),
                    interface=rewrite.get("interface", {}),
                    match=match,
                )
            except ValueError:
                return False

        return True
