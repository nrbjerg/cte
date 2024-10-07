# %% 
from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
from classes.route import Route
from typing import List, Tuple
from classes.node import Node
import numpy as np
from functools import reduce
import random
import time 
import multiprocessing
from sklearn.cluster import KMeans
from core.dijskstra import dijkstra
import matplotlib.pyplot as plt 

def greedy(problem: TOPInstance, routes: List[Route], p: float = 1.0, seed: int = 0) -> List[Route]:
    """Runs a greedy algorithm on the problem instance, with initial routes given"""
    random.seed(seed)

    remaining_distances = [problem.t_max - routes[agent_index].distance for agent_index in range(problem.number_of_agents)]

    visited_nodes = set(sum([route.nodes for route in routes], []))
    unvisited_nodes = set(problem.nodes).difference(visited_nodes) 
    
    # Keep going until we reach the sink, in each of the routes.
    for agent_index in range(problem.number_of_agents):
        while (tail := routes[agent_index].nodes[-1]) != problem.sink: 

            # Compute eligible nodes
            eligible_nodes = []
            candidates = unvisited_nodes.intersection(set(tail.adjacent_nodes))

            # This might be because of the fact that we got stuck in a "corner"
            # instead we will simply check all posible unvisited nodes,
            # for the highest score / distance heuristic.
            if len(candidates) == 0: 
                candidates = unvisited_nodes

            # Only look at the nodes which allows us to subsequently go to the sink.
            for node in unvisited_nodes:
                distance_from_tail = np.linalg.norm(tail.pos - node.pos)
                if distance_from_tail + node.distance_to_sink <= remaining_distances[agent_index]:
                    eligible_nodes.append(node)
            
            # Compute the SDR scores
            sdr_scores = {node.node_id: (node.score / np.linalg.norm(tail.pos - node.pos)) for node in eligible_nodes}
            
            # Pick a random node from the RCL list, if p = 1.0, we are simply using a greedy algorithm
            number_of_nodes = int(np.ceil(len(eligible_nodes) * (1 - p)))
            if number_of_nodes == 0:
                number_of_nodes = 1

            rcl = sorted(eligible_nodes, key = lambda node: sdr_scores[node.node_id], reverse=True)[:number_of_nodes] # TODO
            node_to_append = random.choice(rcl)
            routes[agent_index].add_node(node_to_append) # FIXME: This line causes problems, since the tail is not adjacent to the next node.
            remaining_distances[agent_index] -= np.linalg.norm(tail.pos - node_to_append.pos)

            if node_to_append != problem.sink:
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

def _worker_function(data) -> List[Route]:
    """The worker function which will be used during the next multi processing step, the data tuple contains a seed and the start time."""
    (problem, p, initial_routes, seed, best_candidate_solution, time_budget, start_time) = data
    best_score = sum(route.score for route in best_candidate_solution)
    while (time.time() - start_time) < time_budget:
        candidate_solution = greedy(problem, p = p, routes = initial_routes, seed = seed)
        #candidate_solution = _hill_climbing(candidate_solution, problem)

        if sum(route.score for route in candidate_solution) >= best_score:
            best_candidate_solution = candidate_solution
    
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
            #print(initial_routes[-1].distance, np.linalg.norm(centroid - problem.source.pos), np.linalg.norm(centroid - problem.sink.pos))
        
    else:
        initial_routes = [Route([problem.source]) for _ in range(problem.number_of_agents)]

    baseline_solution = greedy(problem, initial_routes)

    # Runs multilpe iteraitons in parallel
    number_of_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=number_of_cores) as pool:
        arguments = [(problem, p, initial_routes, seed, baseline_solution, time_budget, start_time) 
                     for seed in range(number_of_cores)]
        candidate_solutions = [sol for sol in pool.map(_worker_function, arguments) if sol]
 
    best_candidate_solution = max(candidate_solutions, key=lambda candidate_solution: sum(route.score for route in candidate_solution))
    return best_candidate_solution

if __name__ == "__main__":
    top_instance = load_TOP_instances(needs_plotting = True, neighbourhood_level=1)[0]
    #top_instance.plot()
    print("Running Algorithm")
    #routes = greedy(top_instance, p = 0.7)
    routes = grasp(top_instance, p = 1, time_budget = 10, use_centroids=True)
    print("Finished Algorithm")
    top_instance.plot_with_routes(routes, plot_points=True)
