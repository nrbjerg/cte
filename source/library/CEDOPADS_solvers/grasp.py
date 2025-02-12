# %% 
import time
from typing import Tuple, Set

import matplotlib.pyplot as plt 
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSNode, load_CEDOPADS_instances, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from library.CEDOPADS_solvers.local_search_operators import add_free_visits, hill_climbing
import dubins
from library.CEDOPADS_solvers.greedy import greedy 

def grasp(problem: CEDOPADSInstance, time_budget: float, p: float, utility_function: UtilityFunction) -> CEDOPADSRoute:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()

    best_route = greedy(problem, utility_function, p = 1.0)
    best_score = problem.compute_score_of_route(best_route, utility_function)

    i = 1
    while ((time.time() - start_time) < time_budget):
        candidate_route = greedy(problem, utility_function, p = p)
        candidate_route = hill_climbing(candidate_route, problem, utility_function)

        if (candidate_score := problem.compute_score_of_route(candidate_route, utility_function)) > best_score:
            best_route = candidate_route
            best_score = candidate_score
        
        i += 1

    #return best_route
    return add_free_visits(problem, best_route, utility_function)

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.f.d.b.txt", needs_plotting = True)
    route = grasp(problem_instance, 300, p = 0.95, utility_function=utility_function)
    problem_instance.plot_with_route(route, utility_function)
    plt.show()
    #print(problem_instance.compute_score_of_route(add_free_visits(problem_instance, route, utility_function), utility_function))