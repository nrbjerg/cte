from __future__ import annotations
from dataclasses import dataclass
from classes.node import Node
from functools import cached_property
from typing import List

@dataclass
class Route:
    """Models a route via a starting node and a list of indicies, """
    nodes: List[Node]

    indicies: List[int] # Suppose we are in nodes[i], then nodes[i].adjacent_nodes[indicies[i]] = nodes[i + 1]
    #                     NOTE: used to speed up the computation of the total cost, since we can simply lookup
    #                     the cost of going to the next node, instead of searching for the node.

    @cached_property
    def cost(self) -> float:
        """The cost of the route."""
        total_cost = 0
        for node, index in zip(self.nodes, self.indicies):
            total_cost += node.costs[index]

        return total_cost
            
    @cached_property
    def score(self) -> float:
        """The score of the route."""
        return sum(node.score for node in self.nodes)

    # NOTE: If indicies arent needed in the future, then we don't need to update them (since we assume that the underlying graph is non-directed.)
    def reverse(self, update_indicies: bool = True) -> Route: 
        """Computes the reverse of a route, so if the current route is v_0, v_1, ..., v_n, the reverse will be v_n, v_(n - 1), ..., v_0"""
        nodes = list(reversed(self.nodes))
        if update_indicies:
            indicies = []
            for current_node, following_node in zip(nodes[:-1], nodes[1:]):
                indicies.append([node.node_id for node in current_node.adjacent_nodes].index(following_node.node_id))
            return Route(nodes, indicies)
        else:
            return Route(nodes, self.indicies)

    def __repr__ (self) -> str:
        """Returns a simple string representation of the route."""
        return ",".join([str(node.node_id) for node in self.nodes])