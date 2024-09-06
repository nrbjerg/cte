from dataclasses import dataclass
from node import Node
from typing import List

@dataclass
class Route:
    """Models a route via a starting node and a list of indicies, """
    nodes: List[Node]

    indicies: List[int] # Suppose we are in nodes[i], then nodes[i].adjacent_nodes[indicies[i]] = nodes[i + 1]
    #                     NOTE: used to speed up the computation of the total cost, since we can simply lookup
    #                     the cost of going to the next node, instead of searching for the node.

    @property
    def cost(self) -> float:
        """The cost of the route."""
        total_cost = 0
        for node, index in zip(self.nodes, self.indicies):
            total_cost += node.costs[index]

        return total_cost
            
    @property
    def score(self) -> float:
        """The score of the route."""
        return sum(node.score for node in self.nodes)
