# %% 
import copy
import multiprocessing
import random
import time
from typing import List, Tuple

import numpy as np
from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
from classes.route import Route
from core.dijskstra import dijkstra
from sklearn.cluster import KMeans


def greedy(problem: TOPInstance, routes: List[Route], p: float = 1.0) -> List[Route]:
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
            eligible_nodes = [node for node in unvisited_nodes if (np.linalg.norm(tail.pos - node.pos) + node.distance_to_sink) <= remaining_distances[agent_index]]
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

#def _hill_climbing(initial_routes: List[Route], problem: TOPInstance) -> List[Route]:
#    """Performs hillclimbing, in order to adaptively improve the routes"""
#    improved_routes = []
#    for initial_route in initial_routes:
#        best_route = initial_route
#        remove_operator = True
#        while True:
#            if remove_operator:
#                route, change_in_score, change_in_distance = remove_node(best_route, unvisited_nodes)
#            else:
#                route, change_in_score, change_in_distance = add_node(best_route, unvisited_nodes)
#            
#            if (best_route.score + change_in_score) / (best_route.distance + change_in_distance) > best_route.score / best_route.distance:
#                best_route = route
#            else:
#                if remove_operator
#
#    updated_routes = routes
#                
#    
#    return improved_routes# TODO: Add hill climbing

def _worker_function(data: Tuple[TOPInstance, float, List[Route], int, List[Route], int | float, int, int | None]) -> List[Route]:
    """The worker function which will be used during the next multi processing step, the data tuple contains a seed and the start time."""
    (problem, p, initial_routes, seed, best_candidate_solution, time_budget, start_time, maximum_number_of_iterations) = data
    best_score = sum(route.score for route in best_candidate_solution)
    random.seed(seed)
    iteration_number = 0
    while (time.time() - start_time) < time_budget and (maximum_number_of_iterations is None or iteration_number < maximum_number_of_iterations):
        candidate_solution = greedy(problem, p = p, routes = copy.deepcopy(initial_routes)) # FOR SOME REASON THIS ONLY PRODUCE THE SAME ROUTE?

        if sum(route.score for route in candidate_solution) > best_score:
            best_candidate_solution = candidate_solution
            best_score = sum(route.score for route in candidate_solution)
            print(f"best score found on core {seed}: {best_score}")

        iteration_number += 1

    return best_candidate_solution

def grasp(problem: TOPInstance, time_budget: int, p: float, use_centroids: bool = False) -> List[Route]:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()

    if use_centroids:
        np.random.seed(0); random.seed(0)
        positions = [node.pos for node in problem.nodes[1:-1]]
        # Maybe use score / distance to source node instead?
        weights = [node.score / (np.linalg.norm(problem.source.pos - node.pos) ** 2 + node.distance_to_sink ** 2) for node in problem.nodes[1:-1]] 
        k_means = KMeans(n_clusters = problem.number_of_agents, max_iter = 300).fit(positions, sample_weight = weights)

        # Create routes to the nodes closest to the cluster centers.
        initial_routes = []
        for centroid in k_means.cluster_centers_:
            node_closest_to_centroid = min(problem.nodes, key = lambda node: np.linalg.norm(node.pos - centroid))
            initial_routes.append(dijkstra(problem.nodes, start = problem.source, end = node_closest_to_centroid))
            #plt.scatter(*node_closest_to_centroid.pos, zorder=6, c = "black")
        
    else:
        initial_routes = [Route([problem.source]) for _ in range(problem.number_of_agents)]

    baseline_solution = greedy(problem, copy.deepcopy(initial_routes))
    print(f"Greedy Score: {sum(route.score for route in baseline_solution)}")

    # Runs multilpe iteraitons in parallel
    number_of_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(number_of_cores) as pool:
        arguments = [(problem, p, initial_routes, seed, baseline_solution, time_budget, start_time, None) 
                     for seed in range(1, number_of_cores + 1)]
        candidate_solutions = [sol for sol in pool.map(_worker_function, arguments)]
 
    best_candidate_solution = max(candidate_solutions, key = lambda candidate_solution: sum([route.score for route in candidate_solution]))
    return best_candidate_solution

if __name__ == "__main__":
    top_instance = load_TOP_instances(needs_plotting = True, neighbourhood_level=1)[0]
    #top_instance.plot()
    print("Running Algorithm")
    routes = grasp(top_instance, p = 0.8, time_budget = 180, use_centroids=True)
    print(routes, sum(route.score for route in routes))
    print("Finished Algorithm")
    top_instance.plot_with_routes(routes, plot_points=True)
