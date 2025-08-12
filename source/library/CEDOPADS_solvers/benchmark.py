from typing import Tuple, Iterable, List, Dict

import os 
from library.CEDOPADS_solvers.memetic import GA
from library.CEDOPADS_solvers.sampling_followed_by_exact import solve_sampled_CEDOPADS
from library.CEDOPADS_solvers.samplers import equidistant_sampling, CEDOPADSSampler
import multiprocessing
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, utility_fixed_optical, UtilityFunction, load_CEDOPADS_instances
from classes.data_types import Vector
import datetime
import numpy as np 
import json

def worker_function_self_adaptive_memetic_algorithm (args: Tuple[CEDOPADSInstance, float, int]) -> Tuple[str, List[float], List[float]]:
    """Runs the actual benchmarking of the self-adaptive memetic algorithm on a problem instance."""
    problem_instance, time_budget, number_of_repetitions = args

    scores, run_times = [], []
    for _ in range(number_of_repetitions):
        ga = GA(problem_instance, utility_function, seed=np.random.choice(1024))
        score, run_time = ga.run(time_budget)
        scores.append(score)
        run_times.append(run_time)

    return problem_instance.problem_id, scores, run_times

def benchmark_memetic_algorithm(problem_instances: Iterable[CEDOPADSInstance], log_file: str, time_budget: float, number_of_repetitions: int):
    """Benchmarks the self-adaptive memetic algorithm in a manner which can be disrupted during the computation."""
    path_to_log_file = os.path.join(os.getcwd(), "benchmarks", "CEDOPADS", log_file)
    if os.path.exists(path_to_log_file):
        with open(path_to_log_file, "r") as file:
            json_obj: Dict[str, Dict[str, List[float]]] = json.load(file)
    else:
        json_obj = {}

    problem_instances_which_has_not_already_been_registered = set(problem_instances).difference(json_obj.keys())
    print(f"Running benchmarking the memetic algorithm on {len(problem_instances_which_has_not_already_been_registered)} CEDOPADS instances taking a total of a maximum of {time_budget * number_of_repetitions * len(problem_instances_which_has_not_already_been_registered)} seconds.")

    batch_size = multiprocessing.cpu_count()
    with multiprocessing.Pool() as p:
        for i in range(0, len(problem_instances_which_has_not_already_been_registered), multiprocessing.cpu_count()):
            batch_of_problem_instances = problem_instances_which_has_not_already_been_registered[i:i + batch_size]

            # Get scores from the current batch
            args = [(problem_instance, time_budget, number_of_repetitions) for problem_instance in batch_of_problem_instances]
            outputs = p.map(worker_function_self_adaptive_memetic_algorithm, args)
            for (problem_id, scores, run_times) in outputs:
                json_obj[problem_id] = {"scores": scores, "run times": run_times}

            # Save run data from Genetic algorithm 
            with open(path_to_log_file, "w+") as file:
                json.dump(json_obj, file) 

def benchmark_sampling_followed_by_exact (problem_instances: Iterable[CEDOPADSInstance], log_file: str, sampling_method: CEDOPADSSampler, time_budget: float, number_of_repetitions: int):
    """Benchmarks the self-adaptive memetic algorithm in a manner which can be disrupted during the computation."""
    path_to_log_file = os.path.join(os.getcwd(), "benchmarks", "CEDOPADS", log_file)
    if os.path.exists(path_to_log_file):
        with open(path_to_log_file, "r") as file:
            json_obj: Dict[str, Dict[str, List[float]]] = json.load(file)
    else:
        json_obj = {}

    problem_instances_which_has_not_already_been_registered = set(problem_instances).difference(json_obj.keys())
    print(f"Running sampling -> exact on {len(problem_instances_which_has_not_already_been_registered)} CEDOPADS instances taking a total of a maximum of {time_budget * number_of_repetitions * len(problem_instances_which_has_not_already_been_registered)} seconds.")

    for problem_instance in problem_instances_which_has_not_already_been_registered:
        json_obj[problem_instance.problem_id] = {"obj": [], "gap": [], "run time": []}
        for _ in range(number_of_repetitions):
            _, obj_val, gap, run_time = solve_sampled_CEDOPADS(problem_instance, utility_function, sampling_method, time_budget = time_budget)
            json_obj[problem_instance.problem_id]["obj"].append(obj_val)
            json_obj[problem_instance.problem_id]["gap"].append(gap)
            json_obj[problem_instance.problem_id]["run time"].append(run_time)

        
        # Save new information as well
        with open(path_to_log_file, "w+") as file:
            json.dump(json_obj, file) 

if __name__ == "__main__":
    # Specify utility function
    utility_function = utility_fixed_optical
    utility_function_str = "fixed_optical"

    problem_instances = [CEDOPADSInstance.load_from_file(file_name) for file_name in [
        "p4.4.g.c.a.txt"
    ]]

    benchmark_sampling_followed_by_exact(problem_instances, f"sampling_followed_by_exact_{utility_function_str}.json", equidistant_sampling, time_budget = 3600, number_of_repetitions=1)
    benchmark_memetic_algorithm(problem_instances, f"memetic_{utility_function_str}.json", time_budget = 300, number_of_repetitions = 2)
