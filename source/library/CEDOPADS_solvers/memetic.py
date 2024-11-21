# %%
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, load_CPM_HTOP_instances
from typing import List, Tuple
from library.core.relaxed_dubins import compute_relaxed_dubins_path
from classes.data_types import State
import numpy as np 
import dubins 
import matplotlib.pyplot as plt
import time
import random

def greedy(problem: CEDOPADSInstance, sensing_radius: float, rho: float, t_max: float, p: float = 1.0) -> CEDOPADSRoute:
    """Runs a greedy algorithm on the problem instance, with initial routes given"""
    # Initialization
    remaining_distance = t_max 
    unvisited_nodes = set(problem.nodes)
    distances_from_sink = {}
    for node in unvisited_nodes:
        for idx_of_theta, theta in enumerate(node.thetas):
            distances_from_sink[(node.node_id, idx_of_theta)] = compute_relaxed_dubins_path(node.compute_state(sensing_radius, theta), problem.sink, rho)

    # Figure out the initial node in the route greedily
    route = []
    while True:
        # Compute eligible sdr scores of eligible states
        sdr_scores_of_eligible_states = {}
        for node in unvisited_nodes:
            for idx_of_theta, theta in enumerate(node.thetas):
                q = problem.nodes[node.node_id].compute_state(sensing_radius, theta)

                if len(route) == 0:
                    # NOTE: Remember we are computing the length of the "reversed" dubins path.
                    length_of_dubins_path = compute_relaxed_dubins_path(q.angle_complement(), problem.source, rho)
                else:
                    tail = problem.nodes[route[-1][0]].compute_state(sensing_radius, route[-1][1]) # The last state in the route
                    length_of_dubins_path = dubins.shortest_path(tail.to_tuple(), q.to_tuple(), rho).path_length()
                    
                if length_of_dubins_path < remaining_distance - distances_from_sink[(node.node_id, idx_of_theta)]:
                    sdr_scores_of_eligible_states[(node.node_id, idx_of_theta)] = node.score / length_of_dubins_path

        # If we have no eligible states, simply return the route, since we cannot perform a random greedy choice of the next state.
        if len(sdr_scores_of_eligible_states) == 0:
            return route 
        
        # Compute the reduces candidate list (RCL) and pick a random state as the next state from the list.
        else:
            if p < 1.0:
                number_of_nodes = int(np.ceil(len(sdr_scores_of_eligible_states) * (1 - p)))
            else:
                number_of_nodes = 1
            
            rcl = sorted(sdr_scores_of_eligible_states.keys(), key = lambda info: sdr_scores_of_eligible_states[info], reverse=True)[:number_of_nodes] 
            # NOTE for some reason it coldnt directly chose from RCL
            i = np.random.choice(range(len(rcl))) 
            (node_id, idx_of_theta) = rcl[i]
            route.append((node_id, problem.nodes[node_id].thetas[idx_of_theta]))
            remaining_distance -= problem.nodes[node_id].score / sdr_scores_of_eligible_states[(node_id, idx_of_theta)] # Gets the length since sdr is score / distance.
            unvisited_nodes.remove(problem.nodes[node_id]) 

def generate_random_population(problem_instance: CEDOPADSInstance, n_pop: int, sensing_radius: float, rho: float, t_max: float, p: float = 0.8) -> List[CEDOPADSRoute]:
    """Generates a list of CEDOPADS routes"""
    population = []
    for _ in range(n_pop):
        # Generate a random element based on ideas presented in GRASP using the thetas 
        # to discretise the search space.
        population.append(greedy(problem_instance, sensing_radius, rho, t_max, p = p))

    return population

def compute_penalty(length: float, t_max: float, ratio_of_time_remaining: float) -> float:
    """Computes a penalty for not obeying the distance constraint, based on the remaining time for optimization."""
    if length < t_max:
        return 0
    else:
        return np.exp((length - t_max) / ratio_of_time_remaining)
        
def mutate(problem_instance: CEDOPADSInstance, route: CEDOPADSRoute, temperature: float) -> CEDOPADSRoute:
    """Mutates the route, based on the given parameters."""
    return route

def compute_offspring(problem_instance: CEDOPADSInstance, parents: Tuple[CEDOPADSRoute, CEDOPADSRoute]) -> CEDOPADSRoute:
    """Computes the offspring of two parents."""
    
    return parents[0]

def memetic_algorithm(problem_instance: CEDOPADSInstance, n_pop: int, sensing_radius: float, rho: float, t_max: float, time_budget: float = 60, debug: bool = False) -> CEDOPADSRoute:
    """Runs a memetic algorithm on the problem instance, using a population sized of n_pop."""
    start_time = time.time()
    # 1. Initialize population
    population = generate_random_population(problem_instance, n_pop, sensing_radius, rho, t_max, p = 0.6)
    if debug:
        plt.ion()
    
    # NOTE: At the moment we have created the route such that every route is a candidate
    goat, score_of_goat = None, 0

    while remaining_time := (time.time() - start_time) < time_budget:
        ratio_of_time_remaining = 1 - remaining_time / time_budget

        # Compute the fitnesss of each individual
        scores = np.array([problem_instance.compute_score_of_route(route) for route in population])

        # NOTE: we could maybe allow the routes to have a higher length than t_max, 
        # but penalize them harder and harder as the time budget depletes this could perhaps help the search
        # i have seen this is some paper, which would need to be cited.
        penalties = np.array([compute_penalty(problem_instance.compute_length_of_route(route, sensing_radius, rho), t_max, remaining_time) for route in population])

        # Penalize routes which are to long more and more as time goes on.
        fitnesses = scores - (penalties / (ratio_of_time_remaining ** 2)) 

        # Check if we have a new goat.
        candidates_and_scores = [(route, scores[i]) for i, route in enumerate(population) if penalties[i] == 0]
        (candidate, score_of_candidate) = max(candidates_and_scores, key=lambda tup: tup[1])
        if score_of_candidate > score_of_goat:
            goat = candidate
            score_of_goat = score_of_candidate
            print(f"Found a new GOAT: {goat} with a score of {score_of_goat}")

        # Plot the best routes, which obeys the distance constraint, in a non-blocking way
        if debug: 
            scores_of_candidates = [score for _, score in candidates_and_scores]
            routes = [route for i, (route, _) in enumerate(candidates_and_scores) if (i in np.argpartition(scores_of_candidates, -4)[-4:])]
            problem_instance.plot_with_routes(routes, sensing_radius, rho, show = False)
            plt.draw()
            plt.show()

        # Compute next generation and set a new population
        
        weights = fitnesses - min(fitnesses) * np.ones(n_pop) # TODO: We can probabily get some further improvements by updating this.
        probabilities = weights / sum(weights) if sum(weights) != 0 else (1 / n_pop) * np.ones(n_pop)
        temperature = 0.1 # TODO: Should probabily change with ratio_of_remaining_time.
        new_generation = []
        for _ in range(n_pop):
            parent_indicies = np.random.choice(n_pop, 2, p = probabilities, replace=False)
            parents = tuple(population[i] for i in parent_indicies)
            non_mutated_offspring = compute_offspring(problem_instance, parents)
            new_generation.append(mutate(problem_instance, non_mutated_offspring, temperature))

        population = new_generation

    # Check if we have a new goat in the final generation of the population
    candidates = filter(lambda r: problem_instance.compute_length_of_route(r, sensing_radius, rho) < t_max, population)
    candidate = max(candidates, key=problem_instance.compute_score_of_route)
    if problem_instance.compute_score_of_route(candidate) > score_of_goat:
        return candidate
    else:
        return goat

    # Idea for local search will be to see which nodes along the paths can be added with very minimal cost?

if __name__ == "__main__":
    problem_instance = load_CPM_HTOP_instances(needs_plotting = True)[0]
    n_pop = 100
    sensing_radius = 1
    rho = 1
    t_max = 40
    route = memetic_algorithm(problem_instance, n_pop, sensing_radius, rho, t_max, time_budget = 10, debug=True)
    print(f"GOAT: {route}, with a score of {problem_instance.compute_score_of_route(route)}")

