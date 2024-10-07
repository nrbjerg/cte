from __future__ import annotations
from dataclasses import dataclass
from classes.node import Node
from functools import cached_property
from typing import List
import numpy as np

@dataclass
class Route:
    """Models a route via a starting node and a list of indicies, """
    nodes: List[Node]

    @cached_property
    def distance(self) -> float:
        """The cost of the route."""
        return sum([np.linalg.norm(fst.pos - snd.pos) for fst, snd in zip(self.nodes[1:], self.nodes[:-1])])
            
    @cached_property
    def score(self) -> float:
        """The score of the route."""
        return sum(node.score for node in self.nodes)

    def _reset_cached_properties(self):
        """Resets the cached properties, usefull for modifying the route"""
        self.__dict__.pop('cost', None)
        self.__dict__.pop('score', None)

    # NOTE: If indicies arent needed in the future, then we don't need to update them (since we assume that the underlying graph is non-directed.)
    def reverse(self, update_indicies: bool = True) -> Route: 
        """Computes the reverse of a route, so if the current route is v_0, v_1, ..., v_n, the reverse will be v_n, v_(n - 1), ..., v_0"""
        return Route(list(reversed(self.nodes)))

    def add_node(self, node: Node):
        """Adds a node to the end of the route and computes the correct index."""
        self.nodes.append(node)
        self._reset_cached_properties()

    def __repr__ (self) -> str:
        """Returns a simple string representation of the route."""
        return ",".join(map(str, self.get_node_ids()))

    def get_node_ids (self) -> List[int]:
        """Returns the node ids in the order specified by the route."""
        return [node.node_id for node in self.nodes]