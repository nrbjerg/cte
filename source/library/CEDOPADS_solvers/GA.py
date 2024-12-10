# %%
import random
import time
from typing import List, Tuple, Optional, Callable
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute
from library.core.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle, AngleInterval
from numpy.typing import ArrayLike
from classes.node import Node
import hashlib
from tqdm import tqdm
from itertools import combinations_with_replacement
from scipy.stats import truncnorm
from collections import OrderedDict
import matplotlib.pyplot as plt

class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem: CEDOPADSInstance
    t_max: float
    eta: float
    rho: float
    sensing_radius: float
    mu: int # The number of individuals in each generation

    def __init__ (self, problem: CEDOPADSInstance, t_max, eta, rho, sensing_radius: float, mu: int = 256, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        self.problem = problem
        self.t_max = t_max
        self.eta = eta
        self.rho = rho
        self.sensing_radius = sensing_radius
        self.mu = mu
        self.node_indicies = set(range(len(self.problem.nodes)))

        # Seed RNG for repeatability
        if seed is None:
            hash_of_id = hashlib.sha1(problem.problem_id.encode("utf-8")).hexdigest()[:8]
            seed = int(hash_of_id, 16)
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Make sure that the sink can be reached from the source otherwise there is no point in running the algorithm.
        if (d := np.linalg.norm(self.problem.source - self.problem.sink)) > self.t_max:
            raise ValueError(f"Cannot construct a valid dubins route since ||source - sink|| = {d} is greater than t_max = {self.t_max}")

        # Generate an initial population, which almost satifies the distant constraint, ie. add nodes until
        # the sink cannot be reached within d - t_max units where d denotes the length of the route.
        self.population = []
        for _ in range(self.mu):
            route = []
            while len(route) == 0 or self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) < self.t_max:
                route.append(self.pick_new_visit(route, 0))

            self.population.append(route)

    def pick_new_visit(self, individual: CEDOPADSRoute, ratio_of_time_remaining: float) -> Tuple[int, Angle, Angle]:
        """Picks a new place to visit, according to how much time is left to run the algorithm."""
        # For the time being simply pick a random node to visit FIXME: Should be updated, 
        # maybe we pick them with a probability which is propotional to the distance to the individual?
        blacklist = set(k for (k, _, _) in individual)

        k = random.choice(list(self.node_indicies.difference(blacklist)))
        psi = random.choice(self.problem.nodes[k].intervals).generate_uniform_angle()
        tau = (psi + np.pi + np.random.uniform( - self.eta / 2, self.eta / 2)) % (2 * np.pi)

        return (k, psi, tau)

    # ---------------------------------------- Fitness function ------------------------------------------ 
    def fitness_function(self, individual: CEDOPADSRoute, ratio_of_time_used: float) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        # We wish to add a penalty to penalize routes which are to long
        # and to make the penalty more and more suvere as time is running out.
        length = self.problem.compute_length_of_route(individual, self.sensing_radius, self.rho)
        length_penalty = 1 / (1 + np.exp(-max(length - self.t_max, 0)))

        return 2 * (1 - length_penalty) * self.problem.compute_score_of_route(individual, self.eta)
     
    # -------------------------------------- Recombination Operators -------------------------------------- #
    def crossover(self, parents: List[CEDOPADSRoute], k: int = 4) -> List[CEDOPADSRoute]:
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

                # Remove dublicate visits to the same node, since these do not increase the collected score.
                seen = set()
                idx = 0
                while idx < len(child):
                    if child[idx][0] in seen:
                        child.pop(idx)
                    else:
                        seen.add(child[idx][0])
                        idx += 1

            offspring.append(child)

        return offspring 

    def samir_crossover(self, parents: List[CEDOPADSRoute]) -> List[CEDOPADSRoute]:
        """Performs order crossover, by iteratively selecting two parents and producing their offspring."""
        # Compute offspring for each pair of parents included in the parents list.
        offspring = []
        for pair in combinations_with_replacement(parents, 2):
            
            # Pick indicies for where we swap and do the swapping.
            indicies_and_parent = ((parent, sorted(np.random.choice(len(parent), size=2, replace=False))) for parent in pair)
            
        return offspring

    #--------------------------------------------- Mutation Operators -------------------------------------- #
    def mutate(self, individual: CEDOPADSRoute, ratio_of_time_remaining, p_s, p_id, p_r, q0, q1, scale_modifier_for_psi, scale_for_tau: float) -> CEDOPADSRoute: 
        # Example arguments for p onwards: 0.3, 0.2, 0.6, 0.02, 0.5
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route."""
        u = np.random.uniform(0, 1)
        if u < p_s and len(individual) > 1: 
            # Swap two random visists along the route described by the individual.
            indicies = np.random.choice(len(individual), size = 2, replace = False)
            individual[indicies[0]], individual[indicies[1]] = individual[indicies[1]], individual[indicies[0]]

        # Remove random visists along the route, if posible.
        while np.random.uniform(0, 1) < p_id and len(individual) > 0:
            idx = np.random.choice(len(individual), size = 1)[0]
            individual.pop(idx)

        # Replace visists along the route.
        if len(individual) != 0 and np.random.uniform(0, 1) < p_r:
            idx = np.random.choice(len(individual), size = 1)[0]
            individual[idx] = self.pick_new_visit(individual, ratio_of_time_remaining)

        # Insert a random number of new visits into the route.
        while np.random.uniform(0, 1) < p_id:
            new_visit = self.pick_new_visit(individual, ratio_of_time_remaining)

            if len(individual) == 0:
                individual = [new_visit]

            else:
                idx = np.random.choice(len(individual), size = 1)[0]
                individual.insert(idx, new_visit)

        return self.angle_mutation(individual, q0, q1, scale_for_tau, scale_modifier_for_psi)

    def angle_mutation (self, individual: CEDOPADSRoute, q0, q1, scale_for_tau, scale_modifier_for_psi: float) -> CEDOPADSRoute:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
        assert q0 + q1 <= 1

        for idx, (k, psi, _) in enumerate(individual):
            u = random.uniform(0, 1)
            if u < q0:
                # Compute probabilities of picking an angle intervals based on their "arc lengths"
                # with higher arc lengths having having  a higher probability of getting chosen.
                arc_lengths = []
                for interval in self.problem.nodes[k].intervals:
                    arc_length = interval.b - interval.a if interval.a < interval.b else (2 * np.pi - interval.b) + interval.a
                    arc_lengths.append(arc_length)

                interval_probabilities = np.array(arc_lengths) / sum(arc_lengths)
                interval: AngleInterval = np.random.choice(self.problem.nodes[k].intervals, size = 1, p = interval_probabilities)[0]
                
                # Generate angles from interval
                new_psi =  interval.generate_uniform_angle()

            elif u > q0 and u <= q1 + q0: 
                # NOTE: Core idea is that we use this to "wiggle" the weights to slightly increase the fitness
                
                # Figure out which interval the current angle is located within.
                for interval in self.problem.nodes[k].intervals:
                    if interval.contains(psi):
                        break

                # Set bounds for random number generation NOTE: There are checks for these in the data_types.py file.
                new_psi = interval.generate_truncated_normal_angle(mean = psi, scale_modifier = scale_modifier_for_psi)
                
            else: 
                # NOTE: We mutate the current visit.
                continue 

            # Generate a new tau non-uniformly based on the scale_for_tau argument
            new_tau = truncnorm.rvs(-self.eta / (2 * scale_for_tau), self.eta / (2 * scale_for_tau), 
                                    loc = np.pi + new_psi, scale = scale_for_tau) % (2 * np.pi)
            individual[idx] = (k, new_psi, new_tau)
            
        return individual

    # ---------------------------------------- Parent Selection ----------------------------------------- #
    def windowing(self) -> ArrayLike:
        """Uses windowing to compute the probabilities that each indvidual in the population is chosen for recombination."""
        minimum_fitness = np.min(self.fitnesses)
        modified_fitnesses = self.fitnesses - minimum_fitness
        return modified_fitnesses / np.sum(modified_fitnesses)

    def sigma_secaling(self, c: float = 2) -> ArrayLike:
        """Uses sigma scaling to compute the probabilities that each indvidual in the population is chosen for recombination."""
        mean = np.mean(self.fitnesses)
        std_var = np.std(self.fitnesses)
        modified_fitnesses = np.maximum(self.fitnesses - (mean - c * std_var), 0)
        return modified_fitnesses / np.sum(modified_fitnesses)

    def exponential_ranking(self) -> ArrayLike:
        """Uses expoential ranking to compute the probabilities that each indvidual in the population is chosen for recombination."""
        rankings = np.argsort(self.fitnesses)
        modified_fitnesses = 1 - np.exp(-rankings)
        return modified_fitnesses / np.sum(modified_fitnesses)

    # --------------------------------------- Sampling of parents --------------------------------------- #
    def stochastic_universal_sampling (self, cdf: ArrayLike, m: int) -> List[CEDOPADSRoute]:
        """Implements the SUS algortihm, used to sample m parents from the population."""
        parents = [None for _ in range(m)] 
        i, j = 0, 0
        r = np.random.uniform(0, 1 / m)

        # The core idea behind the algorithm is that instead of doing m-roletewheels to find m parents, 
        # we do a single spin with a roletewheel with m equally spaced indicators (ie. we pick r once and
        # the offset for each of the indicators is 1 / m)
        while i < m:
            while r <= cdf[j]:
                parents[i] = self.population[j]
                r += 1 / m
                i += 1
            
            j += 1

        return parents

    def roulette_wheel (self, cdf: ArrayLike, m: int) -> List[CEDOPADSRoute]:
        """Implements roulette wheel sampling, to sample m parents from the population."""
        parents = []

        # Basically we just sample the population according to the probabilities given indirectly by the CDF
        # which corresponds to spinning m roulette wheels where the lengths of each segment is proportional to
        # the probability of picking each of the individuals of the population.
        for _ in range(m):
            r = np.random.uniform(0, 1)
            i = 0
            while r <= cdf[i] or i == self.mu - 1:
                i += 1
            
            parents.append(self.population[i - 1])
            
        return parents

    # ----------------------------------------- Survivor Selection -------------------------------------- #
    def mu_plus_lambda_selection(self, offspring: List[CEDOPADSRoute], ratio_of_time_used: float) -> List[CEDOPADSRoute]:
        """Merges the newly generated offspring with the generation and selects the best mu individuals to survive until the next generation."""
        fitnesses_of_offspring = np.array(list(map(lambda child: self.fitness_function(child, ratio_of_time_used), offspring)))
        fitnesses_of_parents_and_offspring = np.concatenate((self.fitnesses, fitnesses_of_offspring), axis = 0)

        # Calculate the indicies of the best performing memebers
        indicies_of_new_generation = np.argsort(fitnesses_of_parents_and_offspring)[-self.mu:]
        
        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_offspring[indicies_of_new_generation]

        return [self.population[i] if i < self.mu else offspring[i - self.mu] for i in indicies_of_new_generation]

    def mu_comma_lambda_selection(self, offspring: List[CEDOPADSRoute], ratio_of_time_used: float) -> List[CEDOPADSRoute]:
        """Merges the newly generated offspring with the generation and selects the best mu individuals to survive until the next generation."""
        fitnesses_of_offspring = np.array(list(map(lambda child: self.fitness_function(child, ratio_of_time_used), offspring)))

        # Calculate the indicies of the best performing memebers
        indicies_of_new_generation = np.argsort(fitnesses_of_offspring)[-self.mu:]
        
        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_offspring[indicies_of_new_generation]

        return [offspring[i] for i in indicies_of_new_generation]

    #-------------------------------------------- Main Loop of GA --------------------------------------- #
    def run(self, time_budget: float, m: int, parent_selection_mechanism: Callable[[], List[CEDOPADSRoute]], crossover_mechanism: Callable[[List[CEDOPADSRoute]], List[CEDOPADSRoute]], survivor_selection_mechanism: Callable[[List[CEDOPADSRoute], float], ArrayLike]) -> CEDOPADSRoute:
        """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
        start_time = time.time()

        # Compute fitnesses of each route in the population and associated probabilities using 
        # the parent_selection_mechanism passed to the method, which sould be one of the parent
        # selection methods defined earlier in the class declaration.
        self.fitnesses = [self.fitness_function(individual, 0) for individual in self.population]

        # NOTE: The fitnesses are set as a field of the class so they are accessable within the method
        probabilities = parent_selection_mechanism() 
        cdf = np.add.accumulate(probabilities)

        i = 0
        with tqdm(total=time_budget, desc="GA Loop") as pbar:
            while (elapsed_time := (time.time() - start_time)) < time_budget:
                ratio_of_time_used = elapsed_time / time_budget

                # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
                # and perform k-point crossover using these parents, to create a total of m^k children which is
                # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
                # the angles in order to find new versions of the parents with higher fitness values.
                offspring = []
                for _ in range(self.mu):
                    parents = self.stochastic_universal_sampling(cdf, m = m)
                    
                    # NOTE: If children equal their parents we need to make sure that we still have sufficient variation in the genetic pool.
                    for child in crossover_mechanism(parents): 
                        offspring.append(self.mutate(child, ratio_of_time_used, 0.2, 0.3, 0.2, 0.2, 0.4, 0.02, 0.5))

                # Replace the worst performing individuals based on their fitness values, 
                # however do it softly so that the population increases over time
                # Survivor selection mechanism (Replacement)
                i += 1
                self.population = survivor_selection_mechanism(offspring, ratio_of_time_used)
                #self.population = sorted(offspring + self.population, key = lambda individual: self.fitness_function(individual, ratio_of_time_remaining), reverse=True)[:self.mu] 

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                pbar.n = round(elapsed_time, 1)
                pbar.set_postfix({
                    "Generation": i,
                    "Highest Fitness": max(self.fitnesses),
                    "Average Fitness": np.sum(self.fitnesses) / self.mu
                })
    
        # TODO: Finally try to fix routes which are to long before returning the best route
        permitable_routes = list(filter(lambda route: self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) < self.t_max, self.population))
        return max(permitable_routes, key = lambda route: self.problem.compute_score_of_route(route, self.eta))

if __name__ == "__main__":
    problem: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
    t_max = 200
    rho = 4
    sensing_radius = 3
    eta = np.pi / 5
    ga = GA(problem, t_max, eta, rho, sensing_radius)
    route = ga.run(60, 4, ga.exponential_ranking, ga.crossover, ga.mu_comma_lambda_selection)
    problem.plot_with_route(route, sensing_radius, rho, eta)
    plt.plot()
