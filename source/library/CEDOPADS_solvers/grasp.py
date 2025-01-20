# %% 
import copy
import multiprocessing
import random
import time
from typing import List, Tuple, Set
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute
from library.core.relaxed_dubins import compute_relaxed_dubins_path
from classes.data_types import State
from classes.node import Node
from classes.route import Route
from library.core.dijskstra import dijkstra
from sklearn.cluster import KMeans
from library.CEDOPADS_solvers.greedy import greedy

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

def _worker_function(data: Tuple[TOPInstance, float, List[Route], int, List[Route], int | float, int, int | None]) -> List[Route]:
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

def grasp(problem: CEDOPADSInstance, time_budget: float, t_max: float, rho: float, sensing_radius: float, p: float, use_centroids: bool = False) -> List[Route]:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()

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

#if __name__ == "__main__":
#    problem: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
#    t_max = 200 
#    rho = 2
#    sensing_radius = 2
#    eta = np.pi / 5
#    ga = GA(problem, t_max, eta, rho, sensing_radius, mu = 512, lmbda=512 * 7)
#    print(ga)
#    #cProfile.run("ga.run(60, 3, ga.sigma_scaling, ga.unidirectional_greedy_crossover, ga.mu_comma_lambda_selection, p_c = 1.0, progress_bar = True)", sort = "cumtime")
#    route = ga.run(300, 3, ga.sigma_scaling, ga.unidirectional_greedy_crossover, ga.mu_comma_lambda_selection, p_c = 1.0, progress_bar = True)
#    problem.plot_with_route(route, sensing_radius, rho, eta)
#    plt.show()