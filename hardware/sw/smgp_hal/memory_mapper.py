"""Memory mapper: allocates graph nodes/edges to crossbar tiles.

Handles defragmentation and wear leveling for the memristor crossbar
simulated in hardware.

References:
    Ielmini, D. & Wong, H.-S.P. (2018). "In-memory Computing." Nature Electronics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class MemoryMapper:
    """Maps graph nodes and edges to memristor crossbar tiles.

    The crossbar is organized as TILE_SIZE x TILE_SIZE tiles. Each node
    is mapped to a contiguous row in the crossbar. The mapper handles:
      - Tile allocation
      - Defragmentation when tiles become sparse
      - Wear leveling (simulated via periodic rotation)

    Attributes:
        tile_size: Number of rows/columns per tile.
        num_tiles: Number of available tiles.
        tile_usage: Bitmap of tile usage.
        write_counts: Write count per tile for wear leveling.
    """

    def __init__(self, tile_size: int = 1024, num_tiles: int = 16,
                 hd_dim: int = 10000):
        """Initialize the memory mapper.

        Args:
            tile_size: Rows/columns per crossbar tile.
            num_tiles: Number of crossbar tiles.
            hd_dim: HD vector dimension.
        """
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.hd_dim = hd_dim
        self.tile_usage: List[List[int]] = []  # node_id per slot
        self.write_counts: Dict[int, int] = {}  # tile -> write count
        self._next_addr = 0

    def allocate_node(self, node_id: str) -> Tuple[int, int]:
        """Allocate a crossbar row for a node.

        Args:
            node_id: Node identifier.

        Returns:
            Tuple of (tile_index, row_within_tile).
        """
        tile = self._next_addr // self.tile_size
        row = self._next_addr % self.tile_size

        if tile >= self.num_tiles:
            # Trigger defragmentation
            self._defragment()
            tile = self._next_addr // self.tile_size
            row = self._next_addr % self.tile_size

        self._next_addr += 1
        self.write_counts[tile] = self.write_counts.get(tile, 0) + 1

        if tile < len(self.tile_usage):
            self.tile_usage[tile].append(hash(node_id))
        else:
            self.tile_usage.append([hash(node_id)])

        return (tile, row)

    def allocate_edge(self, source_id: str, target_id: str,
                      relation: str) -> Tuple[int, int]:
        """Allocate crossbar space for an edge.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Relation type.

        Returns:
            Tuple of (tile_index, row_within_tile).
        """
        edge_key = hash((source_id, target_id, relation))
        return self.allocate_node(f"_edge_{edge_key}")

    def _defragment(self) -> None:
        """Compact sparse tiles to free space.

        Moves all entries from partially-filled tiles into a compact
        layout, freeing up fragmented tiles.
        """
        all_entries = []
        for tile_entries in self.tile_usage:
            all_entries.extend(tile_entries)
        self.tile_usage = [all_entries[i:i + self.tile_size]
                          for i in range(0, len(all_entries), self.tile_size)]
        self._next_addr = len(all_entries)

    def wear_level(self) -> None:
        """Rotate tile mapping to distribute write wear.

        Simulates wear leveling by cycling the tile offset.
        """
        if self.tile_usage:
            self.tile_usage.append(self.tile_usage.pop(0))
            self._next_addr = sum(len(t) for t in self.tile_usage)

    def get_address(self, node_id: str) -> Optional[Tuple[int, int]]:
        """Get the physical address of a node.

        Args:
            node_id: Node identifier.

        Returns:
            Tuple of (tile, row) or None.
        """
        node_hash = hash(node_id)
        for tile_idx, entries in enumerate(self.tile_usage):
            for row_idx, entry_hash in enumerate(entries):
                if entry_hash == node_hash:
                    return (tile_idx, row_idx)
        return None

    def utilization(self) -> float:
        """Get crossbar utilization fraction.

        Returns:
            Float between 0 and 1.
        """
        total_slots = self.tile_size * self.num_tiles
        used_slots = sum(len(t) for t in self.tile_usage)
        return used_slots / max(total_slots, 1)
