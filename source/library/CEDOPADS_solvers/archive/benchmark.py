from typing import Tuple, Iterable

import random
import os 
import logging
from library.CEDOPADS_solvers.grasp import GRASP
from library.CEDOPADS_solvers.greedy import greedy
from library.CEDOPADS_solvers.memetic import GA
import multiprocessing
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, utility_fixed_optical, UtilityFunction, load_CEDOPADS_instances
from library.CEDOPADS_solvers.local_search_operators import add_free_visits, get_samples
from classes.data_types import Vector
import datetime
import numpy as np 

def benchmark(problem_instances: Iterable[CEDOPADSInstance], log_file: str, maximal_time_budget_GA: float, maximal_time_budget_sampling_followed_by_exact: float):
    # Run time & score for sampling + exact
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        outputs = p.map(worker_function_GA())
    # Save run data from Genetic algorithm 

# This is what needs to be updates each time.
def worker_function_GA(args: Tuple[int, float, CEDOPADSInstance]) -> Tuple[Vector, str]:
    """The actual function which calls the genenetic algorithm and generates the results"""
    number_of_repetitions, time_budget, problem_instance = args
    utility_function = utility_fixed_optical
    ga = GA(problem_instance, utility_function, mu = 512, lmbda=512 * 7)
    routes = [ga.run(time_budget) for _ in range(number_of_repetitions)]
    
    # Should simply return the results from the algorithm.
    info = f"Memetic {utility_function=}, {time_budget=}, {number_of_repetitions=}"
    return (np.array([problem_instance.compute_score_of_route(add_free_visits(problem_instance, route, utility_function), utility_function) for route in routes]), info)

def worker_function_grasp(args: Tuple[int, float, CEDOPADSInstance]) -> Tuple[Vector, str]:
    """The actual function which calls the GRASP and generates the results"""
    number_of_repetitions, time_budget, problem_instance = args
    utility_function = utility_fixed_optical
    grasp = GRASP(problem_instance, utility_function)
    p = 0.99
    routes = [grasp.run(time_budget, p, get_samples) for _ in range(number_of_repetitions)]
    
    # Should simply return the results from the algorithm.
    info = f"GRASP: {utility_function=}, {p=}, {time_budget=}, {number_of_repetitions=}"
    return (np.array([problem_instance.compute_score_of_route(route, utility_function) for route in routes]), info)

def worker_function_greedy(args: Tuple[int, float, CEDOPADSInstance]) -> Tuple[Vector, str]:
    """The actual function which calls the GRASP and generates the results"""
    _, _, problem_instance = args
    utility_function = utility_fixed_optical
    
    routes = [greedy(problem_instance, utility_function)]
    
    # Should simply return the results from the algorithm.
    info = f"Greedy: {utility_function=}"
    return (np.array([problem_instance.compute_score_of_route(route, utility_function) for route in routes]), info)

def quick_benchmark(k: int = 14, number_of_repetitions: int = 10, time_budget: float = 300.0):
    """Quickly benchmarks the memetic algorithm on a total of m problem instances, using multiprocessing"""
    logger = logging.getLogger(__name__)
    log_file = os.path.join(os.getcwd(), "logs", f"{datetime.date.today()}-{datetime.datetime.now().strftime('%H:%M')}.log")
    logging.basicConfig(filename=log_file, encoding="utf-8", level = logging.INFO)

    random.seed(42)
    problem_instances = random.sample(load_CEDOPADS_instances(), k)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        output = p.map(worker_function_grasp, [(number_of_repetitions, time_budget, problem_instance) for problem_instance in problem_instances])

        #logger.info(f"Problems: {[problem_instance.problem_id for problem_instance in problem_instances]}")
        for i, (result, info) in enumerate(output):
            if i == 0: # No need to repeat our selves.
                logger.info(info)

            logger.info(f"ID: {problem_instances[i].problem_id}, max = {np.max(result):.1f}, min = {np.min(result):.1f}, mean = {np.mean(result):.1f}, std = {np.std(result):.1f}")

if __name__ == "__main__":
    quick_benchmark()