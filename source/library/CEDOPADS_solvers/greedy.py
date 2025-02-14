import matplotlib.pyplot as plt
from typing import List, Tuple, Set

import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSNode, load_CEDOPADS_instances, CEDOPADSRoute, Visit, utility_fixed_optical, UtilityFunction
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from library.CEDOPADS_solvers.local_search_operators import add_free_visits, get_samples
import dubins

def greedy(problem: CEDOPADSInstance, utility_function: UtilityFunction, p: float = 1.0) -> CEDOPADSRoute:
    """Runs a greedy algorithm on the problem instance, with initial routes given"""
    remaining_distance = problem.t_max

    # 1. Pick first visit 
    candidate_visits = {}
    for k in range(len(problem.nodes)):
        for i in range(len(problem.nodes[k].thetas)):
            for j, (psi, tau) in enumerate(get_samples(problem, k, i)):
                q = problem.get_state((k, psi, tau, problem.sensing_radii[1]))
                score = problem.compute_score_of_visit((k, psi, tau, problem.sensing_radii[1]), utility_function)
                length = compute_length_of_relaxed_dubins_path(q.angle_complement(), problem.source, problem.rho)
                candidate_visits[(k, i, j)] = score / length

    # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
    if p < 1.0:
        number_of_nodes = int(np.ceil(len(candidate_visits) * (1 - p)))
    else:
        number_of_nodes = 1

    rcl = [tup[0] for tup in sorted(candidate_visits.items(), key = lambda tup: tup[1], reverse=True)[:number_of_nodes]] 
    k, i, j = rcl[np.random.randint(0, len(rcl))]
    psi, tau = get_samples(problem, k, i)[j]
    route = [(k, psi, tau, problem.sensing_radii[1])]

    # Compute the remaning distance.
    q = problem.get_state(route[-1])
    remaining_distance = problem.t_max - compute_length_of_relaxed_dubins_path(q.angle_complement(), problem.source, problem.rho)

    # Mark the visited node as visited
    unvisited_nodes = {k for k in range(len(problem.nodes)) if k != route[-1][0]}
    
    # 2. Keep going until we reach the sink, in each of the routes.
    while True:
        candidate_visits = {}
        tail = problem.get_state(route[-1])
        for k in unvisited_nodes:
            for i in range(len(problem.nodes[k].thetas)):
                for j, (psi, tau) in enumerate(get_samples(problem, k, i)):
                    q = problem.get_state((k, psi, tau, problem.sensing_radii[1]))
                    score = problem.compute_score_of_visit((k, psi, tau, problem.sensing_radii[1]), utility_function)
                    length = dubins.shortest_path(tail.to_tuple(), q.to_tuple(), problem.rho).path_length()
                    #length = dubins.shortest_path(tail.to_tuple(), q.to_tuple(), problem.rho).path_legnth()
                    if length + compute_length_of_relaxed_dubins_path(q, problem.sink, problem.rho) < remaining_distance:
                        candidate_visits[(k, i, j)] = score / length

        if len(candidate_visits) == 0:
            break

        # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
        if p < 1.0:
            number_of_nodes = int(np.ceil(len(candidate_visits) * (1 - p)))
        else:
            number_of_nodes = 1
        
        rcl = [tup[0] for tup in sorted(candidate_visits.items(), key = lambda tup: tup[1], reverse=True)[:number_of_nodes]] 
        k, i, j = rcl[np.random.randint(0, len(rcl))]
        psi, tau = get_samples(problem, k, i)[j]
        route.append((k, psi, tau, problem.sensing_radii[1]))
        unvisited_nodes.remove(k)
        q = problem.get_state(route[-1])
        remaining_distance -= dubins.shortest_path(tail.to_tuple(), q.to_tuple(), problem.rho).path_length()

    return route

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.f.d.b.txt", needs_plotting = True)
    route = greedy(problem_instance, utility_function, p = 1)
    problem_instance.plot_with_route(route, utility_function)
    augmented_route = add_free_visits(problem_instance, route, utility_function)
    print(problem_instance.compute_score_of_route(augmented_route, utility_function))
    problem_instance.plot_with_route(augmented_route, utility_function, color="tab:green")
    plt.show()
    