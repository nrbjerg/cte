# %%
import random
import time
from typing import List, Tuple, Optional
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute
from library.core.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle, AngleInterval
from classes.node import Node
from scipy.special import binom
import hashlib
from tqdm import tqdm
from itertools import combinations_with_replacement

class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem: CEDOPADSInstance
    t_max: float
    eta: float
    rho: float
    sensing_radius: float
    n_pop: int

    def __init__ (self, problem: CEDOPADSInstance, t_max, eta, rho, sensing_radius: float, n_pop: int = 512, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        # 1. Initialize fields and RNG
        self.problem = problem
        self.t_max = t_max
        self.eta = eta
        self.rho = rho
        self.sensing_radius = sensing_radius
        self.n_pop = n_pop
        self.node_indicies = set(range(len(self.problem.nodes)))

        if seed is None:
            hash_of_id = hashlib.sha1(problem.problem_id.encode("utf-8")).hexdigest()[:8]
            seed = int(hash_of_id, 16)
        
        np.random.seed(seed)
        random.seed(seed)
        
        # 2. Initialize population
        if (d := np.linalg.norm(self.problem.source - self.problem.sink)) > self.t_max:
            raise ValueError(f"Cannot construct a valid dubins route since ||source - sink|| = {d} is greater than t_max = {self.t_max}")

        self.population = []
        for _ in range(self.n_pop):
            route = []
            blacklist = set()
            # We add nodes until we reach the distance constraint, in which case we terminate 
            # the creation of the route and add it to the population.
            while len(route) == 0 or self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) < self.t_max:
                route.append(self.pick_new_visit(route, 0))

            self.population.append(route)

    def fitness_function(self, individual: CEDOPADSRoute, ratio_of_time_remaining: float) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        score = self.problem.compute_score_of_route(individual, self.eta)
        length = self.problem.compute_length_of_route(individual, self.sensing_radius, self.rho)

        # We wish to add a penalty to penalize routes which are to long
        # and to make the penalty more and more suvere as time is running out.
        length_penalty = 1 / (1 + np.exp(max(length - self.t_max, 0))) 
        return 2 * (1 - length_penalty) * (score ** 3 / length)  # TODO: We need to include the ratio of time remaning into the fitness function
     
    # Modeled as k-point crossover where k is the number of parents.
    def crossover(self, parents: List[CEDOPADSRoute], k: int = 2) -> List[CEDOPADSRoute]:
        """Performs k-point crossover between the parents, and creates an offspring."""
        assert k >= 2 and len(parents) >= 2

        # Pick random inidices used to create the k-point crossovers,
        # ie. each parent gets split at the indicies and the subparts,
        # gets rearanged back into a set of children
        indicies = [sorted(np.random.choice(np.arange(0, len(parent)), size = k , replace = False))
                    for parent in parents]
        
        # Generate all of the posible offspring from the given list of indicies for each parent.
        offspring = []
        for parent_indicies in combinations_with_replacement(range(len(parents)), k):
            
            # Generate the child which corresponds to the parent indicies
            for jdx in parent_indicies:
                parent = parents[jdx]
                for idx in range(k):
                    if idx == 0:
                        child = parent[:indicies[jdx][0]]
                    elif idx == k - 1:
                        child.extend(parent[indicies[jdx][k - 2]:])
                    else:
                        child.extend(parent[indicies[jdx][idx]:indicies[jdx][idx + 1]])

            offspring.append(child)

        return offspring 

    def pick_new_visit(self, individual: CEDOPADSRoute, ratio_of_time_remaining: float) -> Tuple[int, Angle, Angle]:
        """Picks a new place to visit, according to how much time is left to run the algorithm."""
        # For the time being simply pick a random node to visit FIXME: Should be updated, 
        # maybe we pick them with a probability which is propotional to the distance to the individual?
        blacklist = set(k for (k, _, _) in individual)

        k = random.choice(list(self.node_indicies.difference(blacklist)))
        psi = random.choice(self.problem.nodes[k].intervals).generate_uniform_angle()
        tau = (psi + np.pi + np.random.uniform( - self.eta / 2, self.eta / 2)) % (2 * np.pi)

        return (k, psi, tau)

    def mutate(self, individual: CEDOPADSRoute, ratio_of_time_remaining: float, prob_of_adding_visit_given_no_removal: float = 0.3, prob_of_performing_uniform_mutation: float = 0.2) -> CEDOPADSRoute: 
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route."""
        # We only wish to remove visits if the route is to long!
        probability_of_removal = ratio_of_time_remaining * max(0, problem.compute_length_of_route(individual, sensing_radius, rho) - self.t_max)

        p = np.random.uniform(0, 1)
        if p < probability_of_removal and len(individual) > 1:
            # TODO: Remove visit with a probability which is propotional with the inverse of the SDR score.
            visit_to_remove = np.random.choice(len(individual))
            mutated_individual = individual[:visit_to_remove] + individual[visit_to_remove + 1:]

        else:
            # Either insert or replace a visit at the generated index
            new_visit = self.pick_new_visit(individual, ratio_of_time_remaining)
            
            # NOTE: in order to be able to replace / insert a visit we must have an individual containing one or more visits
            if len(individual) > 0: 
                index = np.random.choice(len(individual)) 

                if p < probability_of_removal * (1 - prob_of_adding_visit_given_no_removal): 
                    # NOTE: this replacement works sort of like a random reseting.
                    mutated_individual = individual[:index] + [new_visit] + individual[index + 1:] 

                else:
                    mutated_individual = individual[:index] + [new_visit] + individual[index:] # NOTE: Insert
            
            else:
                mutated_individual = [new_visit]

        # It is probabily disiarable to use a non-uniform mutation operator on the angles which stays in the current angle intervals
        # and have a non-zero probability perhaps close to 50% of generating a random new angle in another angle interval if the node
        # k has more than one permitable angle interval.
        for idx, (k, psi, tau) in enumerate(mutated_individual):
            if random.uniform(0, 1) < prob_of_performing_uniform_mutation:

                # Compute probabilities of picking an angle intervals based on their "arc lengths"
                # with higher arc lengths having having  a higher probability of getting chosen.
                arc_lengths = []
                for interval in self.problem.nodes[k]:
                    arc_length = interval.b - interval.a if interval.a < interval.b else (2 * np.pi - interval.b) + interval.a
                    arc_lengths.append(arc_length)

                interval_probabilities = np.array(arc_lengths) / sum(arc_lengths)
                interval: AngleInterval = np.random.choice(self.problem.nodes[k].intervals, size = 1, p = interval_probabilities)
                
                # Generate angles from interval
                new_psi = interval.generate_uniform_angle()
                new_tau = (new_psi + np.pi + np.random.uniform( - self.eta / 2, self.eta / 2)) % (2 * np.pi)

                # Mutate the given allele.
                mutated_individual[idx] = (k, new_psi, new_tau)
            
        return mutated_individual

    def run(self, time_budget: float, m: int, k: int = 2, debug: bool = False, mutation_probability: float = 0.6) -> CEDOPADSRoute:
        """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
        start_time = time.time()

        i = 0
        with tqdm(total=time_budget, desc="GA Loop") as pbar:
            while (elapsed_time := (time.time() - start_time)) < time_budget:
                ratio_of_time_remaining = 1 - (elapsed_time) / time_budget

                #print(f"Currently at iteration: {i} with a total of {ratio_of_time_remaining * 100:.1f}% time remaning.")

                # Compute fitnesses of each route in the population
                fitnesses = np.array([self.fitness_function(individual, ratio_of_time_remaining) for individual in self.population])
                average_fitness = 1 / self.n_pop * np.sum(fitnesses)

                # Epsilon is just below the minimum fitness meaning every individual have a non-zero 
                # probability of getting picked for selection, however epsilon also ensures that the
                # probability of picking the individual with the worst fitness is not extremely small.
                epsilon = np.min(fitnesses) * 0.9
                probabilities = 1 / (np.sum(fitnesses) - self.n_pop * epsilon) * (fitnesses - epsilon)

                # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
                # and perform k-point crossover using these parents, to create a total of m^k children which is
                # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
                # the angles in order to find new versions of the parents with higher fitness values.
                offspring = []
                for _ in range(self.n_pop):
                    candidate_parents = [self.population[idx] for idx in np.random.choice(len(self.population), size=m, replace=False, p = probabilities)]
                    selected_parents = sorted(candidate_parents, key = lambda route: self.fitness_function(route, ratio_of_time_remaining), reverse=True)[:m]
                    
                    # NOTE: If children equal their parents we need to make sure that we still have sufficient variation in the genetic pool.
                    for child in self.crossover(selected_parents, k): 
                        if np.random.uniform(0, 1) < mutation_probability:
                            offspring.append(self.mutate(child, ratio_of_time_remaining))
                        else:
                            offspring.append(child)

                # Replace the worst performing individuals based on their fitness values, 
                # however do it softly so that the population increases over time
                # Survivor selection mechanism (Replacement)
                i += 1
                self.population = sorted(offspring + self.population, key = lambda individual: self.fitness_function(individual, ratio_of_time_remaining), reverse=True)[:self.n_pop + 8 * i] 
                self.n_pop += 8 * i

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                pbar.n = round(elapsed_time, 1)
                pbar.set_postfix({
                    "Generation": i,
                    "Highest Fitness": max(fitnesses),
                    "Average Fitness": average_fitness
                })
    
        # TODO: Finally try to fix routes which are to long before returning the best route
        permitable_routes = list(filter(lambda route: self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) < self.t_max, self.population))
        return max(permitable_routes, key = lambda route: self.problem.compute_score_of_route(route, self.eta))

if __name__ == "__main__":
    problem = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
    t_max = 200
    rho = 4
    sensing_radius = 2
    eta = np.pi / 5
    route = GA(problem, t_max, eta, rho, sensing_radius).run(time_budget = 60, m = 5, debug=True)
    print(route)
    problem.plot_with_route(route, sensing_radius, rho, eta)

# %%
