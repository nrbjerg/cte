from source.classes.TOP_instances import TOPInstance, Node
from typing import Callable, List, Tuple
from source.classes.Route import Route, compute_length_of_route

def compute_candidates(current_route: Route, top: TOPInstance, current_route_lengt: float = None):
    """Returns a list of candidate nodes (i.e. nodes )"""
    if current_route_length is None:
        current_route_length = compute_length_of_route(current_route, top)

# NOTE: callable arguments should be of the form (node_under_investigation, (current_end_node, underlying_TOP_instance))
def greedy(top: TOPInstance, criteria: Callable[[Node, Tuple[Node, TOPInstance]], float]) -> Route:
    """Performs a greedy search using a given criteria."""
    route, current_length = [0], 0
    while len(candidates := compute_candidates(route, top, current_length = current_length)) != 0:
        pass 

    return route