from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Tuple 
from classes.data_types import Position
import numpy as np 

@dataclass(eq=False)
class Node:
    """Stores information about a node."""
    node_id: int
    adjacent_nodes: List[Node]
    pos: Position
    score: float
    distance_to_sink: float | None = None
    size: float | None = None # The size of the node in plots, if we need to plot it.

    def __hash__(self) -> int:
        """Used for hashing in order to create sets of nodes"""
        return self.node_id
    
    def __eq__(self, other: Node) -> bool:
        return self.node_id == other.node_id

    def __ne__(self, other: Node) -> bool:
        if not isinstance(other, Node):
            raise ValueError(f"Tried to compare an object of type {type(other)}, to the node: {self.__repr__()}")
        return self.node_id != other.node_id
    
    def __repr__(self):
        return f"({self.node_id}: {float(self.pos[0]), float(self.pos[1])} {self.score})"

def compute_cost (v: Node, w: Node) -> float:
    """Computes the euclidian distance between a pair of nodes."""
    return np.sqrt(np.square(v.pos[0] - w.pos[0]) + np.square(v.pos[1] - w.pos[1]))

def populate_costs (nodes: List[Node]):
    """Pupulate the scores, note that this function does not impose the asusmption that we have a undirected graph."""
    for idx, node in enumerate(nodes):
        for adjacent_node in node.adjacent_nodes:
            nodes[idx].costs.append(compute_cost(node, adjacent_node)) 
