# %% 
import copy
import multiprocessing
import random
import time
from typing import List, Tuple, Set
import os 
import matplotlib.pyplot as plt

import numpy as np
from classes.problem_instances.cpm_htop_instances import CPM_HTOP_Instance, load_CPM_HTOP_instances
from classes.node import Node
from classes.route import Route
from core.dijskstra import dijkstra
from core.interception.euclidian_intercepter import EuclidianInterceptionRoute
from sklearn.cluster import KMeans


def greedy(problem: CPM_HTOP_Instance, routes: List[Route], p: float = 1.0) -> List[Route]:
    """Runs a greedy algorithm on the problem instance, with initial routes given"""
    remaining_distances = [problem.t_max - routes[agent_index].distance for agent_index in range(problem.number_of_agents)]

    visited_nodes = set(sum([route.nodes for route in routes], []))
    unvisited_nodes = set(problem.nodes).difference(visited_nodes).difference([problem.sink])
    
    # Keep going until we reach the sink, in each of the routes.
    for agent_index in range(problem.number_of_agents):
        while (tail := routes[agent_index].nodes[-1]) != problem.sink:

            # Compute eligible nodes
            candidates = unvisited_nodes.intersection(set(tail.adjacent_nodes))

            # This might be because of the fact that we got stuck in a "corner"
            # instead we will simply check all posible unvisited nodes,
            # for the highest score / distance heuristic.
            if len(candidates) == 0: 
                candidates = unvisited_nodes

            # Only look at the nodes which allows us to subsequently go to the sink.
            eligible_nodes = [node for node in unvisited_nodes if 
                              (np.linalg.norm(tail.pos - node.pos) + node.distance_to_sink) <= remaining_distances[agent_index]]
            if len(eligible_nodes) == 0:
                routes[agent_index].add_node(problem.sink)
            
            else:
                sdr_scores = {node.node_id: (node.score / np.linalg.norm(tail.pos - node.pos)) for node in eligible_nodes}

                # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
                if p < 1.0:
                    number_of_nodes = int(np.ceil(len(eligible_nodes) * (1 - p)))
                else:
                    number_of_nodes = 1
                
                rcl = sorted(eligible_nodes, key = lambda node: sdr_scores[node.node_id], reverse=True)[:number_of_nodes] # TODO
                node_to_append = np.random.choice(rcl)
                routes[agent_index].add_node(node_to_append)
                remaining_distances[agent_index] -= np.linalg.norm(tail.pos - node_to_append.pos)

                unvisited_nodes.remove(node_to_append) 
            
    return routes

def _add_node(route: Route, t_max: float, blacklist: Set[Node]) -> Tuple[Route, float, float]:
    """Finds the best non-visited nodes which can be added to the route, while keeping the distance sufficiently low."""
    remaining_distance = t_max - route.distance 
    best_candidate, best_idx_to_insert_candidate = None, None
    best_sdr, best_change_in_distance = 0, None 
    
    # Find the best candidate which can be added to the route
    for idx, (node0, node1) in enumerate(zip(route.nodes[:-1], route.nodes[1:]), 1):
        candidates = set(node0.adjacent_nodes).difference(blacklist).difference(route.nodes)
        distance_between_node0_and_node1 = np.linalg.norm(node0.pos - node1.pos)

        # Find the best candidate to add between node0 and node1
        for candidate in candidates:
            distance_through_candidate = np.linalg.norm(node0.pos - candidate.pos) + np.linalg.norm(candidate.pos - node1.pos)
            change_in_distance = distance_through_candidate - distance_between_node0_and_node1 
            if change_in_distance == 0:
               best_candidate = candidate
               best_idx_to_insert_candidate, best_change_in_distance = idx, change_in_distance
               break # We have found a node which we can freely add

            if (change_in_distance < remaining_distance) and (candidate.score / change_in_distance > best_sdr):
                best_candidate = candidate
                best_sdr = candidate.score / change_in_distance
                best_idx_to_insert_candidate, best_change_in_distance = idx, change_in_distance
        # If the inner for looop terminates normally, we simply continue otherwise we break out of the current for loop as well 
        else: 
            continue
        break

    # Add the candidate to the route
    if best_candidate:
        new_route = Route(route.nodes[:best_idx_to_insert_candidate] + [best_candidate] + route.nodes[best_idx_to_insert_candidate:])
        return (best_candidate, new_route, best_change_in_distance, best_candidate.score)
    
    else:
        return (None, route, 0, 0) 

def _remove_node(route: Route) -> Tuple[Node | None, Route, float, float]:
    """Tries to remove the worst performing node (measured via the SDR) from the route"""
    worst_sdr = np.inf
    idx_of_worst_candidate = None

    for idx, (node0, node1, node2) in enumerate(zip(route.nodes[:-2], route.nodes[1:-1], route.nodes[2:]), 1):
        # Check if node1 should be skiped.
        change_in_distance = np.linalg.norm(node0.pos - node2.pos) - np.linalg.norm(node0.pos - node1.pos) - np.linalg.norm(node1.pos - node2.pos)
        change_in_score = -node1.score
        if change_in_distance == 0:
            continue
        else:
            sdr = change_in_score / change_in_distance
            if sdr < worst_sdr:
                worst_change_in_distance = change_in_distance
                worst_change_in_score = change_in_score
                idx_of_worst_candidate = idx
                worst_sdr = sdr

    if idx_of_worst_candidate:
        new_route = Route(route.nodes[:idx_of_worst_candidate] + route.nodes[idx_of_worst_candidate + 1:])
        return (route.nodes[idx_of_worst_candidate], new_route, worst_change_in_distance, worst_change_in_score)
    else:
        return (None, route, 0, 0)
        
def _hill_climbing(initial_solution: List[Route], t_max: float) -> List[Route]: # TODO: Maybe we could benefit from 2-opt
    """Performs hillclimbing, in order to improve the routes, in the initial solution."""
    updated_routes = []
    for idx, initial_route in enumerate(initial_solution):

        # We will never add the same nodes to two distinct routes.
        blacklist = set(sum([route.nodes for route in updated_routes], []) + sum([route.nodes for route in initial_solution[idx:]], []))
        best_route = initial_route

        # Remove nodes while we increase the SDR metric of the route.
        while True:
            removed_node, route, change_in_distance, change_in_score = _remove_node(best_route)
            if removed_node in blacklist:
                blacklist.remove(removed_node)

            if (best_route.score + change_in_score) / (best_route.distance + change_in_distance) > best_route.score / best_route.distance:
                best_route = route
            else: 
                break

        # Add as many nodes as posible to the route, in order to increase the score.
        while (info := _add_node(best_route, t_max, blacklist=blacklist))[1] != best_route:
            best_route = info[1]
            if info[0]:
                blacklist.add(info[0])

            else: 
                break

        updated_routes.append(best_route)
                        
    return updated_routes # TODO: Add hill climbing

def _worker_function(data: Tuple[CPM_HTOP_Instance, float, List[Route], int, List[Route], int | float, int, int | None]) -> List[Route]:
    """The worker function which will be used during the next multi processing step, the data tuple contains a seed and the start time."""
    (problem, p, initial_routes, seed, best_candidate_solution, time_budget, start_time, maximum_number_of_iterations) = data
    best_score = sum(route.score for route in best_candidate_solution)

    np.random.seed(seed) 
    random.seed(seed)  
    iteration = 0
    while (((time.time() - start_time) < time_budget) and ((maximum_number_of_iterations is None) or (iteration < maximum_number_of_iterations))):

        candidate_solution = greedy(problem, p = p, routes = copy.deepcopy(initial_routes)) 
        candidate_solution = _hill_climbing(candidate_solution, problem.t_max)

        if sum(route.score for route in candidate_solution) > best_score:
            best_candidate_solution = candidate_solution
            best_score = sum(route.score for route in candidate_solution)

        iteration += 1

    return best_candidate_solution

def grasp(problem: CPM_HTOP_Instance, time_budget: int, p: float, use_centroids: bool = False, maximum_number_of_iterations: int | None = None) -> List[Route]:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()

    if use_centroids:
        # Seed for deterministic results.
        np.random.seed(0) 
        random.seed(0)

        positions = [node.pos for node in problem.nodes[1:-1]]
        weights = [node.score / (np.linalg.norm(problem.source.pos - node.pos) ** 2 + node.distance_to_sink ** 2) for node in problem.nodes[1:-1]] 
        k_means = KMeans(n_clusters = problem.number_of_agents, max_iter = 300).fit(positions, sample_weight = weights)

        # Create routes to the nodes closest to the cluster centers.
        initial_routes = []
        for centroid in k_means.cluster_centers_:
            node_closest_to_centroid = min(problem.nodes, key = lambda node: np.linalg.norm(node.pos - centroid))
            initial_routes.append(dijkstra(problem.nodes, start = problem.source, end = node_closest_to_centroid))
        
    else:
        initial_routes = [Route([problem.source]) for _ in range(problem.number_of_agents)]

    baseline_solution = greedy(problem, copy.deepcopy(initial_routes))

    # Runs multilpe iteraitons in parallel
    number_of_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(1) as pool:
        arguments = [(problem, p, initial_routes, seed, baseline_solution, time_budget, start_time, maximum_number_of_iterations) 
                     for seed in range(1, number_of_cores + 1)]
        candidate_solutions = [sol for sol in pool.map(_worker_function, arguments)]
 
    best_candidate_solution = max(candidate_solutions, key = lambda candidate_solution: sum([route.score for route in candidate_solution]))
    return best_candidate_solution

if __name__ == "__main__":
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "CPM_HTOP") 
    cpm_htop_instance = CPM_HTOP_Instance.load_from_file("p4.3.f.0.txt", needs_plotting=True)
    print("Running Algorithm")
    routes = grasp(cpm_htop_instance, p = 0.7, time_budget = 10, use_centroids = True)
    print("Finished Algorithm")
    euclidian_interception_route = EuclidianInterceptionRoute(cpm_htop_instance.source.pos, [0, 2, 1, 0, 2, 1, 0], routes, 1.6, waiting_time=1)
    cpm_htop_instance.plot_with_routes(routes, plot_points=True, show=False)
    euclidian_interception_route.plot()
