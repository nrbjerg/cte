import numpy as np 
from library.core.interception.euclidian_intercepter import compute_euclid_interception
from classes.problem_instances.cpm_rtop_instances import CPM_RTOP_Instance
from typing import List, Tuple
from dataclasses import dataclass
from classes.data_types import Vector, Route
import matplotlib.pyplot as plt 
import time 


@dataclass
class Individual:
    """Stores the information regarding an individual."""
    uav_routes: List[Route]
    cpm_route: List[Tuple[int, float]]

class Baseline_GA:
    """A baseline genetic algorithm implementation for the CPM-RTOP."""

    def __init__(self, problem: CPM_RTOP_Instance):
        self.problem = problem

    # --------------------------------------- Fitness Function ------------------------------------------ #
    def fitness_function(self, individual: Individual) -> float:
        pass 

    # --------------------------------------- Sampling of parents --------------------------------------- #
    def stochastic_universal_sampling (self, cdf: Vector, m: int) -> List[int]:
        """Implements the SUS algortihm, used to sample the indices of m parents from the population."""
        indices = [None for _ in range(m)] 
        i, j = 0, 0
        r = np.random.uniform(0, 1 / m)

        # The core idea behind the algorithm is that instead of doing m-roletewheels to find m parents, 
        # we do a single spin with a roletewheel with m equally spaced indicators (ie. we pick r once and
        # the offset for each of the indicators is 1 / m)
        while i < m:
            while r <= cdf[j]:
                indices[i] = j 
                r += 1 / m
                i += 1
            
            j += 1

        return indices

    # ----------------------------------------- Survivor Selection -------------------------------------- #
    def mu_comma_lambda_selection(self, offspring: List[Individual]) -> List[Individual]:
        """Picks the mu offspring with the highest fitnesses to populate the the next generation, note this is done with elitism."""
        fitnesses_of_offspring = np.array(list(map(lambda child: self.fitness_function(child), offspring)))

        # Calculate the indices of the best performing memebers
        indices_of_new_generation = np.argsort(fitnesses_of_offspring)[-self.mu:]

        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_offspring[indices_of_new_generation]
        return [offspring[i] for i in indices_of_new_generation]

    def run(self, time_budget: float = 300):
        """Runs the baseline genetic algorithm."""
        start_time = time.time()
        while (time.time() - start_time) < time_budget:
            pass 
            

if __name__ == "__main__":
    first_cpm_RTOP_instance = CPM_RTOP_Instance.load_from_file("p4.2.a.0.txt", needs_plotting=True)
    ga = Baseline_GA(first_cpm_RTOP_instance)

    best_plan = ga.run()
    first_cpm_RTOP_instance.plot_with_plan(best_plan)