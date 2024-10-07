# %% 
from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
from classes.route import Route
from typing import List, Tuple
import numpy as np
from functools import reduce
import random
import time 
import multiprocessing

def greedy(problem: TOPInstance, p: float = 1.0, use_centroids: bool = True, seed: int = 0) -> List[Route]:
    """Runs a greedy algorithm on the problem instance"""
    random.seed(seed)

    routes = [Route([problem.source]) for _ in range(problem.number_of_agents)]
    remaining_distances = [problem.t_max for _ in range(problem.number_of_agents)]

    if use_centroids:
        # TODO:
        pass 

    #visited_nodes = reduce(lambda a, b: a.union(b), map(lambda route: set(route.nodes), routes))
    unvisited_nodes = set(problem.nodes).difference(set([problem.source])) #visited_nodes)
    
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
    
def _worker_function(data) -> List[Route]:
    """The worker function which will be used during the next multi processing step, the data tuple contains a seed and the start time."""
    (problem, p, use_centroids, seed, baseline_score, time_budget, start_time) = data
    best_score = baseline_score
    best_candidate_solution_found = None
    while (time.time() - start_time) < time_budget:
        candidate_solution = greedy(problem, p = p, use_centroids = use_centroids, seed = seed)
        # TODO: Add hill climbing

        if sum(route.score for route in candidate_solution) > best_score:
            best_candidate_solution_found = candidate_solution
    
    return best_candidate_solution_found

def grasp(problem: TOPInstance, time_budget: int, p: float, use_centroids: bool = False) -> List[Route]:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()
    #routes = [Route([problem.source], []) for _ in range(problem.number_of_agents)]

    baseline_solution = greedy(problem, use_centroids=use_centroids)
    baseline_score = sum(route.score for route in baseline_solution)
    print(f"Greedy Score: {baseline_score}")

    number_of_cores = multiprocessing.cpu_count()
    print(f"Number of cores: {number_of_cores}")


    # Runs multiple intances of 
    with multiprocessing.Pool(processes=number_of_cores) as pool:
        arguments = [(problem, p, use_centroids, seed, baseline_score, time_budget, start_time) 
                     for seed in range(number_of_cores)]
        candidate_solutions = [sol for sol in pool.map(_worker_function, arguments) if sol]

    if len(candidate_solutions) == 0:
        return baseline_solution
    
    else:
        best_candidate_solution = max(candidate_solutions, key=lambda candidate_solution: sum(route.score for route in candidate_solution))
        if sum(route.score for route in best_candidate_solution) > baseline_score:
            return best_candidate_solution
        else:
            return baseline_solution

if __name__ == "__main__":
    top_instance = load_TOP_instances(needs_plotting = True, neighbourhood_level=2)[0]
    #top_instance.plot()
    print("Running Algorithm")
    #routes = greedy(top_instance, p = 0.7)
    routes = grasp(top_instance, p = 1, time_budget = 60, use_centroids=False)
    print("Finished Algorithm")
    top_instance.plot_with_routes(routes, plot_points=True)
