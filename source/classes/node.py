from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Tuple 
import numpy as np 

@dataclass
class Node:
    """Stores information about a node."""
    node_id: int
    adjacent_nodes: List[Node]
    costs: List[float]
    pos: Tuple[float, float]
    score: float
    size: float | None = None # The size of the node in plots, if we need to plot it.

    def __hash__(self) -> int:
        """Used for hashing in order to create sets of nodes"""
        return self.node_id

def compute_cost (v: Node, w: Node) -> float:
    """Computes the euclidian distance between a pair of nodes."""
    return np.sqrt(np.square(v.pos[0] - w.pos[0]) + np.square(v.pos[1] - w.pos[1]))

def populate_costs (nodes: List[Node]):
    """Pupulate the scores, note that this function does not impose the asusmption that we have a undirected graph."""
    for idx, node in enumerate(nodes):
        for adjacent_node in node.adjacent_nodes:
            nodes[idx].costs.append(compute_cost(node, adjacent_node)) 
