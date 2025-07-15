from __future__ import annotations
from dataclasses import dataclass
from classes.node import Node
from functools import cached_property
from classes.data_types import Position
from typing import List
import numpy as np

@dataclass
class Route:
    """Models a route via a starting node and a list of indicies, """
    nodes: List[Node]
    speed: float = 1.0 # Should remained fixed at one for CPM-RTOP

    #@cached_property
    @property
    def distance(self) -> float:
        """The cost of the route."""
        return sum([np.linalg.norm(fst.pos - snd.pos) for fst, snd in zip(self.nodes[1:], self.nodes[:-1])])

    @property
    def visit_times(self) -> List[float]:
        """Computes the times when the nodes are visisted"""
        distances = [np.linalg.norm(fst.pos - snd.pos) for fst, snd in zip(self.nodes[1:], self.nodes[:-1])]
        visit_times = [0]
        for distance in distances:
            visit_times.append(visit_times[-1] + distance / self.speed)
        return visit_times
            
    #@cached_property
    @property
    def score(self) -> float:
        """The score of the route."""
        return sum(node.score for node in self.nodes)

    def _reset_cached_properties(self):
        """Resets the cached properties, usefull for modifying the route"""
        self.__dict__.pop('cost', None)
        self.__dict__.pop('score', None)

    def reverse(self) -> Route: 
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

    def get_position(self, time: float) -> Position:
        """Computes the agents position at time t."""
        if time < 0:
            raise ValueError("Cannot go back in time! ;)")

        remaining_time = time
        for fst, snd in zip(self.nodes[:-1], self.nodes[1:]):
            time_used_to_traverse_edge = np.linalg.norm(fst.pos - snd.pos) / self.speed
            if remaining_time <= time_used_to_traverse_edge:
                propotion_between_fst_and_snd = remaining_time / time_used_to_traverse_edge
                return fst.pos * (1 - propotion_between_fst_and_snd) + snd.pos * propotion_between_fst_and_snd
            else:
                remaining_time -= time_used_to_traverse_edge

        return self.nodes[-1].pos # Otherwise we are at the sink.
