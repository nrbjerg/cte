from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, load_CEDOPADS_instances
from library.CEDOPADS_solvers.GA import GA
import multiprocessing
from typing import Tuple, Dict, Any
import random
import numpy as np
import csv
from pqdm.processes import pqdm

PROBLEMS = load_CEDOPADS_instances()[:5]
TIME_BUDGET = 60
T_MAX = 200
ETA = np.pi / 2
RHO = 2
SENSING_RADIUS = 2
TOTAL_NUMBER_OF_TESTS = multiprocessing.cpu_count() 

def run_GA_instance(hyper_parameters: Dict[Any, Any]) -> Tuple[Dict[Any, Any], float, float]:
    """Runs a genetic algorithm instance with the given hyper parameters over a set of problems"""
    best_fitnesses = []
    for problem in PROBLEMS:
        ga = GA(problem, T_MAX, ETA, RHO, SENSING_RADIUS, hyper_parameters["mu"])
        route = ga.run(TIME_BUDGET, hyper_parameters["m"], getattr(ga, hyper_parameters["parent_selection_method"]), ga.multi_index_greedy_crossover,
                       getattr(ga, hyper_parameters["survivor_selection_method"]), progress_bar = False)
        best_fitnesses.append(problem.compute_score_of_route(route, ETA))

    hyper_parameters.update({"mean": round(np.mean(best_fitnesses), 2), "var": round(np.var(best_fitnesses), 2)})
    return hyper_parameters

def random_parameter_sweep():
    """Performs the parameter sweep."""
    HYPER_PARAMETER_RANGES = {
        "p_s": [0.1, 0.2, 0.3],
        "p_i": [0.1, 0.2, 0.3],
        "p_r": [0.1, 0.2, 0.3],
        "q": [0.1, 0.2, 0.3],
        "m": [2, 3, 4], # NOTE: It must be the case that k <= m, and hence it does not make sense to include it directly in the hyper parameter ranges.
        "n": [1, 2, 3],
        "c_s": [1, 2, 4],
        "c_f": [1, 2, 4],
        "mu": [128, 512, 2048],
        #"parent_selection_method": ["sigma_scaling", "windowing", "exponential_ranking"],
        "survivor_selection_method": ["mu_comma_lambda_selection", "mu_plus_lambda_selection"],
    }
    
    # Sample the hyper the space of hyperparameters, in order to test different 
    hyper_parameter_samples = []
    for i in range(TOTAL_NUMBER_OF_TESTS):
        sample = {}
        for k, v in HYPER_PARAMETER_RANGES.items():
            sample[k] = random.choice(v)
            if k == "m":
                sample["k"] = random.choice(range(2, sample[k] + 1))

        hyper_parameter_samples.append(sample)

    results = pqdm(hyper_parameter_samples, run_GA_instance, n_jobs = multiprocessing.cpu_count())
    with open("random_parameter_sweep_results.csv", "w+", newline='') as file:
        fc = csv.DictWriter(file, fieldnames = results[0].keys()) 
        fc.writeheader()
        fc.writerows(results)

random_parameter_sweep()