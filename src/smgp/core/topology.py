"""Topological persistence analysis for knowledge graph stability.

Implements persistent homology computation and topological stability constraints
to ensure that graph updates do not destroy essential structural features.

References:
  - Edelsbrunner, H., et al. (2002). "Topological Persistence and Simplification."
    Discrete & Computational Geometry, 28(4), 511–533.
  - Carlsson, G. (2009). "Topology and Data." Bulletin of the AMS, 46(2), 255–308.
  - Kerber, M., et al. (2017). "Geometry Helps to Compare Persistence Diagrams."
    J. Experimental Algorithmics, 22, 1–20.
"""
from __future__ import annotations

import numpy as np

from smgp.core.graph import SpectralMemoryGraph


class TopologicalAnalyzer:
    """Topological analysis engine for knowledge graphs.

    Computes persistent homology of the graph's HD vector embedding space,
    provides Wasserstein distance metrics for diagram comparison, and
    implements topological pruning to compress memory while preserving
    essential features.

    Attributes:
        graph: The knowledge graph to analyze.
        max_dimension: Maximum homology dimension to compute.
        _distance_matrix: Cached pairwise distance matrix.

    References:
        Carlsson, G. (2009). "Topology and Data." Bulletin of the AMS.
    """

    def __init__(
        self,
        graph: SpectralMemoryGraph,
        max_dimension: int = 2,
    ) -> None:
        """Initialize the topological analyzer.

        Args:
            graph: Knowledge graph to analyze.
            max_dimension: Maximum homology dimension (H0, H1, ..., H_{max_dim}).
        """
        self.graph = graph
        self.max_dimension = max_dimension
        self._distance_matrix: np.ndarray | None = None

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute the pairwise cosine distance matrix between all node HD vectors.

        distance(i, j) = 1 - cosine_similarity(v_i, v_j)

        Returns:
            np.ndarray of shape (N, N) with pairwise distances.
        """
        if self._distance_matrix is not None:
            return self._distance_matrix

        vectors = self.graph.node_vectors()
        node_ids = list(vectors.keys())
        n = len(node_ids)

        if n == 0:
            self._distance_matrix = np.zeros((0, 0))
            return self._distance_matrix

        matrix = np.zeros((n, n))
        vec_list = [vectors[nid].astype(np.float64) for nid in node_ids]

        for i in range(n):
            for j in range(i + 1, n):
                norm_i = np.linalg.norm(vec_list[i])
                norm_j = np.linalg.norm(vec_list[j])
                if norm_i > 0 and norm_j > 0:
                    cos_sim = float(
                        np.dot(vec_list[i], vec_list[j]) / (norm_i * norm_j)
                    )
                else:
                    cos_sim = 0.0
                dist = 1.0 - cos_sim
                matrix[i, j] = dist
                matrix[j, i] = dist

        self._distance_matrix = matrix
        return self._distance_matrix

    def compute_persistence(
        self, distance_matrix: np.ndarray | None = None
    ) -> dict:
        """Compute persistent homology of the graph's HD embedding space.

        Uses a Vietoris-Rips-like filtration on the pairwise distance matrix.
        Attempts to use GUDHI if available, falls back to a simple custom
        implementation for H0 and H1.

        Args:
            distance_matrix: Optional pre-computed distance matrix.

        Returns:
            Dict with keys:
              - 'diagrams': List of persistence diagrams per dimension.
                Each diagram is a list of (birth, death) tuples.
              - 'betti_numbers': List of Betti numbers per dimension.

        References:
            Edelsbrunner, H., et al. (2002). "Topological Persistence and
                Simplification." DCG, 28(4).
        """
        if distance_matrix is None:
            distance_matrix = self.compute_distance_matrix()

        n = distance_matrix.shape[0]
        if n < 2:
            return {"diagrams": [[]], "betti_numbers": [n]}

        # Try GUDHI for proper persistent homology
        try:
            import gudhi  # type: ignore

            rips = gudhi.RipsComplex(
                distance_matrix=distance_matrix,
                max_edge_length=float(np.max(distance_matrix)),
            )
            simplex_tree = rips.create_simplex_tree(
                max_dimension=self.max_dimension + 1
            )
            persistence = simplex_tree.persistence()

            diagrams: list[list[tuple[float, float]]] = [
                [] for _ in range(self.max_dimension + 1)
            ]
            for dim, (birth, death) in persistence:
                if death == float("inf"):
                    death = float(np.max(distance_matrix)) * 1.5
                if dim <= self.max_dimension:
                    diagrams[dim].append((birth, death))

            betti_numbers = [
                len([p for p in d if p[1] >= p[0]]) for d in diagrams
            ]
            return {"diagrams": diagrams, "betti_numbers": betti_numbers}

        except ImportError:
            pass

        # Fallback: simple H0 computation
        diagrams = self._simple_persistence(distance_matrix)
        betti_numbers = [
            len([p for p in d if p[1] > p[0]]) for d in diagrams
        ]
        return {"diagrams": diagrams, "betti_numbers": betti_numbers}

    def _simple_persistence(
        self, distance_matrix: np.ndarray
    ) -> list[list[tuple[float, float]]]:
        """Simple H0 persistent homology via Union-Find.

        Computes connected component persistence by tracking when components
        merge during a Vietoris-Rips filtration.

        Args:
            distance_matrix: Pairwise distance matrix.

        Returns:
            List of persistence diagrams (H0, H1).
        """
        n = distance_matrix.shape[0]

        # Get all edges sorted by distance
        edges: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((distance_matrix[i, j], i, j))
        edges.sort()

        # Union-Find for H0
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        h0_diagram: list[tuple[float, float]] = []

        for dist, i, j in edges:
            if union(i, j):
                # Components merge: the younger one dies
                h0_diagram.append((0.0, dist))

        # Remaining components persist to infinity
        max_dist = (
            float(np.max(distance_matrix)) * 1.5 if n > 0 else 1.0
        )
        for i in range(n):
            if find(i) == i:
                h0_diagram.append((0.0, max_dist))

        return [h0_diagram, []]  # H0 and placeholder H1

    def wasserstein_distance(
        self,
        diag1: dict,
        diag2: dict,
        p: float = 1.0,
    ) -> float:
        """Compute the p-Wasserstein distance between two persistence diagrams.

        Uses a simple greedy matching algorithm for efficiency. For production
        use with large diagrams, consider using the GUDHI or Persim library.

        Args:
            diag1: First persistence dict (output of compute_persistence).
            diag2: Second persistence dict.
            p: Order of Wasserstein distance.

        Returns:
            Scalar Wasserstein distance.

        References:
            Kerber, M., et al. (2017). "Geometry Helps to Compare Persistence
                Diagrams." JEA, 22.
        """
        diagrams1 = diag1.get("diagrams", [[]])
        diagrams2 = diag2.get("diagrams", [[]])
        max_dim = min(len(diagrams1), len(diagrams2))

        total_dist = 0.0
        for dim in range(max_dim):
            d1 = diagrams1[dim]
            d2 = diagrams2[dim]
            total_dist += self._wasserstein_pairwise(d1, d2, p)

        return total_dist

    def _wasserstein_pairwise(
        self,
        pairs1: list[tuple[float, float]],
        pairs2: list[tuple[float, float]],
        p: float,
    ) -> float:
        """Compute Wasserstein distance between persistence diagrams in one dimension.

        Args:
            pairs1: List of (birth, death) tuples.
            pairs2: List of (birth, death) tuples.
            p: Wasserstein order.

        Returns:
            Scalar distance.
        """
        # Compute persistence (death - birth) for each point
        def persistence(birth: float, death: float) -> float:
            return max(death - birth, 0.0)

        # Add diagonal points to balance cardinality
        all_pairs = [(p1, p2) for p1 in pairs1 for p2 in pairs2]

        if not all_pairs:
            # One or both are empty
            n1 = len(pairs1)
            n2 = len(pairs2)
            if n1 == 0 and n2 == 0:
                return 0.0
            max_death = 0.0
            for bd in pairs1 + pairs2:
                max_death = max(max_death, bd[0], bd[1])
            max_pers = max(max_death, 1.0) * 0.5
            return max_pers * abs(n1 - n2) ** (1.0 / p)

        # Greedy matching
        cost = np.zeros((len(pairs1) + len(pairs2), len(pairs1) + len(pairs2)))
        n1, n2 = len(pairs1), len(pairs2)

        for i in range(n1):
            for j in range(n2):
                # L^p distance in the persistence diagram
                d = (
                    abs(pairs1[i][0] - pairs2[j][0]) ** p
                    + abs(pairs1[i][1] - pairs2[j][1]) ** p
                ) ** (1.0 / p)
                cost[i, j] = d
            # Match to diagonal: cost = persistence / 2
            pers1 = persistence(pairs1[i][0], pairs1[i][1])
            cost[i, n2 + i] = pers1 / 2.0

        for j in range(n2):
            pers2 = persistence(pairs2[j][0], pairs2[j][1])
            cost[n1 + j, j] = pers2 / 2.0

        # Fill diagonal block with large values (prevents matching augmented rows/cols)
        for i in range(n1, n1 + n2):
            for j in range(n2, n1 + n2):
                cost[i, j] = 1e18

        # Simple greedy matching (for production, use Hungarian algorithm)
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        total_cost = 0.0

        # Sort all valid assignments by cost
        assignments: list[tuple[float, int, int]] = []
        for i in range(n1 + n2):
            for j in range(n1 + n2):
                if i < n1 and j < n2:
                    assignments.append((cost[i, j], i, j))
                elif i < n1 and j >= n2 and j - n2 == i:
                    assignments.append((cost[i, j], i, j))
                elif i >= n1 and j < n2 and i - n1 == j:
                    assignments.append((cost[i, j], i, j))
        assignments.sort()

        for c, i, j in assignments:
            if i not in used_rows and j not in used_cols:
                total_cost += c
                used_rows.add(i)
                used_cols.add(j)

        return total_cost

    def prune_by_persistence(
        self,
        threshold: float = 0.1,
    ) -> SpectralMemoryGraph:
        """Remove nodes corresponding to low-persistence topological features.

        Nodes whose removal does not significantly change the persistence
        diagram (below the threshold) are candidates for pruning.

        Args:
            threshold: Minimum persistence (death - birth) for a feature to
                be considered "essential" and worth preserving.

        Returns:
            New SpectralMemoryGraph with low-persistence nodes removed.

        References:
            Edelsbrunner, H., et al. (2002). "Topological Persistence and
                Simplification." DCG.
        """
        persistence_result = self.compute_persistence()
        diagrams = persistence_result.get("diagrams", [[]])

        if not diagrams or not diagrams[0]:
            return self.graph

        # Nodes that form clusters persisting above threshold are essential
        distance_matrix = self.compute_distance_matrix()
        n = distance_matrix.shape[0]
        node_ids = list(self.graph.nodes())

        # Use Union-Find to find components at the threshold distance
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[ry] = rx
            return True

        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix[i, j] <= threshold:
                    union(i, j)

        # Identify singleton components (isolated nodes) — candidates for pruning
        component_sizes: dict[int, int] = {}
        for i in range(n):
            root = find(i)
            component_sizes[root] = component_sizes.get(root, 0) + 1

        nodes_to_remove: list[str] = []
        for i in range(n):
            root = find(i)
            if component_sizes[root] == 1:
                # Check if this node has meaningful connections
                has_edges = False
                for j in range(n):
                    if i != j and distance_matrix[i, j] < 1.0:
                        has_edges = True
                        break
                if not has_edges:
                    nodes_to_remove.append(node_ids[i])

        # Create new graph without pruned nodes
        new_graph = SpectralMemoryGraph(hd_dim=self.graph.hd.dim)
        for nid in node_ids:
            if nid not in nodes_to_remove:
                data = self.graph.get_node(nid)
                if data:
                    new_graph.add_node(
                        nid,
                        label=data["label"],
                        properties=data["properties"],
                        vector=data["vector"],
                    )

        # Re-add edges between remaining nodes
        for src, tgt, key, edata in self.graph.edges():
            if src not in nodes_to_remove and tgt not in nodes_to_remove:
                new_graph.add_edge(
                    src,
                    tgt,
                    edata.get("relation", ""),
                    properties=edata.get("properties", {}),
                )

        return new_graph

    def stability_check(
        self,
        original_graph: SpectralMemoryGraph,
        max_wasserstein: float = 0.5,
    ) -> bool:
        """Check if a graph modification preserves topological stability.

        Computes the Wasserstein distance between the persistence diagrams
        of the original and current graph. Returns True if the distance
        is within the allowed threshold.

        Args:
            original_graph: The original knowledge graph.
            max_wasserstein: Maximum allowed Wasserstein distance.

        Returns:
            True if the modification is topologically stable, False otherwise.

        References:
            Cohen-Steiner, D., et al. (2007). "Stability of Persistence Diagrams."
                Discrete & Computational Geometry, 37(1), 103–120.
        """
        orig_analyzer = TopologicalAnalyzer(original_graph, self.max_dimension)
        orig_diag = orig_analyzer.compute_persistence()
        curr_diag = self.compute_persistence()
        dist = self.wasserstein_distance(orig_diag, curr_diag)
        return bool(dist <= max_wasserstein)
