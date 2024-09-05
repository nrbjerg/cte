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

def compute_distance (v: Node, w: Node) -> float:
    """Computes the euclidian distance between a pair of nodes."""
    return np.sqrt(np.square(v.pos[0] - w.pos[0]) + np.square(v.pos[1] - w.pos[1]))