from classes.top_instances import TOPInstance
from classes.node import Node, populate_costs
from typing import List, Dict
from classes.route import Route
from dataclasses import dataclass, field
import numpy as np

# TODO: Could probably be done more efficiently using a priority queue,  however the ones 
#       that i have found, does not allow the user to update the priorities of the nodes.
def reversed_dijkstra(nodes: List[Node], target_id: int = 1) -> Dict[int, Route]:
    """Computes the shortest paths to the node with the target id for each node in the TOP instance."""
    # 1. + 2. Mark all nodes as unvisited and assign the current best known distance from the 
    smallest_costs = {node.node_id: 0 if node.node_id == target_id else np.inf for node in nodes}
    fastest_routes = {target_id: Route([nodes[target_id]], [])}

    unvisited = set([node.node_id for node in nodes])
    while len(unvisited) > 0:
        # Pick the node with the smallests costs 
        current_id = None
        for node_id in unvisited:
            if current_id is None or smallest_costs[node_id] < smallest_costs[current_id]:
                current_id = node_id
            
        current_node = nodes[current_id] 
        indicies = fastest_routes[current_id].indicies
        nodes_in_route = fastest_routes[current_id].nodes

        for index, adjacent_node in enumerate(current_node.adjacent_nodes):
            if adjacent_node.node_id not in unvisited:
                continue

            if adjacent_node.node_id not in fastest_routes.keys() or smallest_costs[current_id] + current_node.costs[index] < smallest_costs[adjacent_node.node_id]:
                fastest_routes[adjacent_node.node_id] = Route(nodes_in_route + [adjacent_node], indicies + [index])
                smallest_costs[adjacent_node.node_id] = smallest_costs[current_id] + current_node.costs[index]

        unvisited.remove(current_id)
    
    return {node_id: route.reverse() for node_id, route in fastest_routes.items()}

if __name__ == "__main__":
    nodes = [
        Node(0, [], [], (0, 0), 1),
        Node(1, [], [], (9, 9), 1),
        Node(2, [], [], (3, 1), 1),
        Node(3, [], [], (3, 3), 1),
        Node(4, [], [], (0, 3), 1),
        Node(5, [], [], (3, 5), 1),
        Node(6, [], [], (6, 2), 1),
        Node(7, [], [], (10, 3), 1),
    ]

    nodes[0].adjacent_nodes = [nodes[2], nodes[4]]
    nodes[1].adjacent_nodes = [nodes[5], nodes[7]]
    nodes[2].adjacent_nodes = [nodes[0], nodes[3]]
    nodes[3].adjacent_nodes = [nodes[2], nodes[4], nodes[6]]
    nodes[4].adjacent_nodes = [nodes[0], nodes[3], nodes[5]]
    nodes[5].adjacent_nodes = [nodes[4], nodes[1]]
    nodes[6].adjacent_nodes = [nodes[3], nodes[7]]
    nodes[7].adjacent_nodes = [nodes[6], nodes[1]]
    
    populate_costs(nodes)

    routes = reversed_dijkstra(nodes)
    print(routes)
