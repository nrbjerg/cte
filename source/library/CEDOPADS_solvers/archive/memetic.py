# %%
import dubins
from copy import deepcopy
import random
import time
from typing import List, Optional, Callable, Set, Tuple
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
import logging
from classes.data_types import State, Angle,AreaOfAcquisition, Matrix, Position, Vector
from numpy.typing import ArrayLike
import hashlib
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import truncnorm
from library.CEDOPADS_solvers.local_search_operators import remove_visit, greedily_add_visits_while_posible, add_free_visits
from dataclasses import dataclass

@dataclass()
class Individual:
    route: CEDOPADSRoute
    states: List[State]
    distances: List[float]
    scores: List[float]

    @property()
    def fitness(self) -> float:
        """Returns the score of the individual, which is computed by summing the scores"""
        return np.sum(self.scores)


class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem_instance: CEDOPADSInstance
    utility_function: UtilityFunction
    mu: int # Population Number
    lmbda: int # Number of offspring
    number_of_nodes: int
    node_indicies: Set[int]
    node_indicies_array: Vector

    def __init__ (self, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, mu: int = 256, lmbda: int = 256 * 7, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        self.problem_instance = problem_instance
        self.utility_function = utility_function
        self.lmbda = lmbda
        self.mu = mu
        self.number_of_nodes = len(self.problem_instance.nodes)
        self.node_indicies = set(range(self.number_of_nodes))
        self.node_indicies_arrray = np.array(range(self.number_of_nodes))
        
        # Seed RNG for repeatability
        if seed is None:
            hash_of_id = hashlib.sha1(problem_instance.problem_id.encode("utf-8")).hexdigest()[:8]
            seed = int(hash_of_id, 16)
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Make sure that the sink can be reached from the source otherwise there is no point in running the algorithm.
        if (d := np.linalg.norm(self.problem_instance.source - self.problem_instance.sink)) > self.problem_instance.t_max:
            raise ValueError(f"Cannot construct a valid dubins route since ||source - sink|| = {d} is greater than t_max = {self.problem_instance.t_max}")

    def initialize_population(self):
        """Initializes the population randomly."""
        blacklist = set(k for (k, _, _, _) in individual)

        k = random.choice(list(self.node_indicies.difference(blacklist)))
        psi = random.choice(self.problem_instance.nodes[k].AOAs).generate_uniform_angle() # TODO:
        tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
        r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

        return (k, psi, tau, r)

    def pick_new_visit_based_on_sdr(self, individual: CEDOPADSRoute, i: int, j: int) -> Visit:
        """Generates a visit, based on the distance from the straight line segment between node i and node j."""
        self.sdrs_for_overwrite[:] = self.sdr_tensor[i][j][:]
        #distances = self.distance_tensor[i][j] 
        self.mask_for_overwrite[:] = self.default_mask[:]
        for (k, _, _, _) in individual:
            self.mask_for_overwrite[k] = False
            #self.distances_for_overwrite[k] = np.max(self.distances_for_overwrite)

        weights = self.sdrs_for_overwrite[self.mask_for_overwrite]
        probs =  weights / np.sum(weights)
        k = np.random.choice(self.node_indicies_arrray[self.mask_for_overwrite], p = probs)
        psi = random.choice(self.problem_instance.nodes[k].AOAs).generate_uniform_angle()
        tau = (psi + np.pi + np.random.uniform( - self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
        r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

        return (k, psi, tau, r)
                
    def pick_new_visit_based_on_sdr_and_index(self, individual: CEDOPADSInstance, idx: int) -> Visit:
        """Picks a new visit based on the ''neighbours'' of the visit as well as the euclidian distance between
           the line segment between the nodes, and the new node, that is the candidate to be added."""
        if idx == 0:
            return self.pick_new_visit_based_on_sdr(individual, individual[0][0], self.number_of_nodes)
        elif idx == len(individual) - 1:
            return self.pick_new_visit_based_on_sdr(individual, individual[-1][0], self.number_of_nodes + 1)
        else:
            return self.pick_new_visit_based_on_sdr(individual, individual[idx][0], individual[idx + 1][0])

    # ---------------------------------------- Fitness function ------------------------------------------ 
    def fitness_function(self, individual: CEDOPADSRoute) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        return self.problem_instance.compute_score_of_route(individual, self.utility_function) 
     
    # -------------------------------------- Recombination Operators -------------------------------------- #
    def unidirectional_greedy_crossover(self, parents: List[CEDOPADSRoute], scores: List[List[float]], states: List[List[State]], k: int = 2, n: int = 2) -> List[CEDOPADSRoute]:
        """Performs a unidirectional crossover, by iteratively selecting two parents and producing their offspring, in a greedy fashion starting from the source."""
        # Compute offspring for each pair of parents included in the parents list.
        offspring = []

        number_of_candidates = k * n
        # NOTE: these repeatedly gets overwritten each time a new greedy choice is made.
        sdr_of_visits = np.zeros(number_of_candidates)
        distances = np.zeros(number_of_candidates)

        for indicies in combinations(range(len(parents)), k):
            pool = [parents[index] for index in indicies]
            
            for dominant_parent in pool:
                # Skip the dominant parent if it is empty.
                if len(dominant_parent) == 0:
                    continue 

                # Keep track of the current index in each of the parents.
                child = [dominant_parent[0]]
                pointer = 1
                seen = np.zeros(self.number_of_nodes)
                seen[child[0][0]] = 1

                # Perform a greedy choice of the next visit in the route based on the sdr score, 
                # from the final state in the route to the next candidate state.
                q = self.problem_instance.nodes[child[0][0]].get_state(child[0][3], child[0][1], child[0][2])
                total_distance = compute_length_of_relaxed_dubins_path(q.angle_complement() , self.problem_instance.source, self.problem_instance.rho)
                while total_distance + compute_length_of_relaxed_dubins_path(q, self.problem_instance.sink, self.problem_instance.rho) < self.problem_instance.t_max:

                    candidates = []
                    for i, (index, parent) in enumerate(zip(indicies, pool)):
                        # Make sure that every route in the pool has an appropriate length and that we have  already visited the greedily chosen node.
                        for j, l in enumerate(range(pointer, pointer + n)):
                            if l >= len(parent) or seen[parent[l][0]] == 1:
                                #new_visit = self.pick_new_visit_based_on_sdr_and_index(child, len(child) - 1)
                                new_visit = self.pick_new_visit(child)
                                distances[i * n + j] = self.problem_instance.compute_distance_between_state_and_visit(q, new_visit) 
                                sdr_of_visits[i * n + j] = self.problem_instance.compute_score_of_visit(new_visit, self.utility_function) / distances[i * n + j]
                                candidates.append(new_visit)

                            else:
                                candidates.append(parent[l])
                                distances[i * n + j] = dubins.shortest_path(q.to_tuple(), states[index][l].to_tuple(), self.problem_instance.rho).path_length()
                                sdr_of_visits[i * n + j] = scores[index][l] / distances[i * n + j] 
                    
                    # Chose the next visit greedily, based on the sdr score.
                    # NOTE: Argmax seems to work alot better than chosing the next with a random probability,
                    # however this can be tested using c, since a higher c yields results more like if we had 
                    # used argmax to choose i.
                    #probs = sdr_of_visits / np.sum(sdr_of_visits)
                    #if np.isnan(probs).any(): 
                    #    # There has been some sort of error in the calculations and we have 
                    #    # no choice but to chose a random candidate, uniformly.
                    #    print("caught")
                    #    i = np.random.choice(number_of_candidates)

                    #else:
                    #    i = np.random.choice(number_of_candidates, p = probs) 
                    
                    i = np.argmax(sdr_of_visits) 
                    total_distance += distances[i]
                    child.append(candidates[i])

                    seen[child[-1][0]] = 1
                    q = self.problem_instance.nodes[child[-1][0]].get_state(child[-1][3], child[-1][1], child[-1][2])
                    pointer += 1

                offspring.append(child[:-1])

        return offspring

    def distance_preserving_crossover (self, parents: List[CEDOPADSRoute], scores: List[List[float]], states: List[List[State]], k: int = 2, n: int = 2) -> List[CEDOPADSRoute]:
        """Performs the distance preserving crossover between each pair of parents."""
        offspring = []

        for (paternal, meternal) in combinations(parents, 2):
            # Find fragments which lie within the paternal and meternal individuals.
            fragments = []

            paternal_node_ids = [paternal[idx][0] for idx in range(len(paternal))]
            meternal_node_ids = [meternal[idx][0] for idx in range(len(meternal))]

            idx = 0
            m_paternal = len(paternal)
            m_meternal = len(meternal)
            indicies_of_visits_not_in_meternal = set(range(0, m_meternal))
            while idx < m_paternal:

                # Find the length of the fragments
                try:
                    jdx = meternal_node_ids.index(paternal_node_ids[idx])
                    for frag_length in range(1, m_meternal - jdx):
                        if idx + frag_length >= m_paternal or paternal_node_ids[idx + frag_length] != meternal_node_ids[jdx + frag_length]:
                            break
                    else:
                        frag_length = 1

                    # Find the fragments and remove the visited verticeis from the nodes not visited within the meternal individual
                    fragments.append(paternal[idx:idx + frag_length] if np.random.uniform(0, 1) < 0.5 else meternal[jdx:jdx + frag_length])
                    for i in range(jdx, jdx + frag_length):
                        indicies_of_visits_not_in_meternal.remove(i)

                    idx += frag_length

                except ValueError:
                    fragments.append([paternal[idx]])
                    idx += 1


            for jdx in indicies_of_visits_not_in_meternal:
                fragments.append([meternal[jdx]])

            # Recombine fragments greedily (maybe stochastically.)
            scores = np.array([self.problem_instance.compute_score_of_route(fragment, self.utility_function) for fragment in fragments])
            lengths = np.array([sum(self.problem_instance.compute_lengths_of_route_segments(fragment)[1:-1]) for fragment in fragments])

            lenghts_of_initial_segments = np.array([compute_length_of_relaxed_dubins_path(self.problem_instance.get_state(fragment[0]).angle_complement(),
                                                                                          self.problem_instance.source, self.problem_instance.rho) for fragment in fragments])

            idx = np.argmax(scores / (lengths + lenghts_of_initial_segments))
            chosen_fragments: List[CEDOPADSRoute] = [fragments[idx]] 
            remaining_distance_budget = self.problem_instance.t_max - (lenghts_of_initial_segments[idx] + lengths[idx])
            scores[idx] = 0
            while (remaining_distance_budget < compute_length_of_relaxed_dubins_path(self.problem_instance.get_state(chosen_fragments[-1][-1]), self.problem_instance.sink, self.problem_instance.rho)):
                idx = np.argmax(scores / lengths)
                chosen_fragments.append(fragments[idx])
                remaining_distance_budget -= (lengths[idx])
                scores[idx] = 0

            offspring.append(sum(chosen_fragments[:-1], []))

        return offspring

    #--------------------------------------------- Mutation Operators -------------------------------------- 

    # TODO: we could look at the pheno type which corresponds to the individdual (i.e. what does the actual route look like.)
    #       when inserting new visits into the routes, this could speed up the convergence rate.
    def mutate(self, individual: CEDOPADSRoute, p_s: float, p_i: float, p_r: float, q: float) -> CEDOPADSRoute: 
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route.""" 
        # Replace visists along the route.
        for i in range(len(individual)):
            if np.random.uniform(0, 1) < p_r:
                individual[i] = self.pick_new_visit_based_on_sdr_and_index(individual, i)

        # Insert a random number of new visits into the route.
        i = 0
        while i < len(individual) and len(individual) < self.number_of_nodes:
            if np.random.uniform(0, 1) < p_i:
                # Generate a visit, based on where in the route it should be inserted,
                # note that self.number_of_nodes and self.number_of_nodes + 1, coresponds
                # to the source and sink nodes respectively, when passed as arguments in the
                # pick_new_visit_based_on_distance method.
                individual.insert(i, self.pick_new_visit_based_on_sdr_and_index(individual, i))  
                i += 2
            else:
                i += 1

        if np.random.uniform(0, 1) < p_s and len(individual) >= 2: 
            # Swap two random visists along the route described by the individual.
            indicies = np.random.choice(len(individual), size = 2, replace = False)
            individual[indicies[0]], individual[indicies[1]] = individual[indicies[1]], individual[indicies[0]]

        return self.angle_mutation(individual, q)

    # TODO: Mayor performance gains to be had here by storing the lengths of the path segments.
    def fix_length(self, individual: CEDOPADSRoute) -> Tuple[CEDOPADSRoute, float]: 
        """Removes visits from the individual until it obeys the distance constraint."""
        # NOTE: These are stored in order to make it easier to look them up later.
        distance = self.problem_instance.compute_length_of_route(individual)
        
        # Save the inverse of the SDR scores of each visit in the route. 
        #weights = [1 / self.sdr_tensor[self.number_of_nodes][individual[1][0]][individual[0][0]]]
        #for i in range(1, len(individual) - 1):
        #    weight = 1 / self.sdr_tensor[individual[i - 1][0]][individual[i + 1][0]][individual[i][0]]
        #    weights.append(weight)

        #weights.append(1 / self.sdr_tensor[self.number_of_nodes + 1][individual[-2][0]][individual[-1][0]])
        #sum_of_weights = sum(weights)


        # TODO: This can be optimized heavily if we keep track of the lengths underway instead of recomputing them each time.
        while distance > self.problem_instance.t_max: 
            individual, change_in_distance, _ = remove_visit(individual, self.problem_instance, self.utility_function)
            distance += change_in_distance

            #weights = np.empty(len(individual))
            #weights[0] = 1 / self.sdr_tensor[self.number_of_nodes][individual[1][0]][individual[0][0]]
            #for i in range(1, len(individual) - 1):
            #    weights[i] = 1 / self.sdr_tensor[individual[i - 1][0]][individual[i + 1][0]][individual[i][0]]

            #weights[-1] = 1 / self.sdr_tensor[self.number_of_nodes + 1][individual[-2][0]][individual[-1][0]]
            
            #index = np.random.choice(len(individual), p = [weight / np.sum(weights) for weight in weights]) 
            
            ## Update the distance and weights based on the index being poped.
            #if index == 0:
            #    distance += (compute_length_of_relaxed_dubins_path(states[1].angle_complement(), self.problem_instance.source, self.problem_instance.rho) -
            #                 dubins.shortest_path(tuples[0], tuples[1], self.problem_instance.rho).path_length() -
            #                 compute_length_of_relaxed_dubins_path(states[0].angle_complement(), self.problem_instance.source, self.problem_instance.rho))
            #elif index < len(individual) - 1:
            #    distance += (dubins.shortest_path(tuples[index - 1], tuples[index + 1], self.problem_instance.rho).path_length() -
            #                 dubins.shortest_path(tuples[index - 1], tuples[index], self.problem_instance.rho).path_length() -
            #                 dubins.shortest_path(tuples[index], tuples[index + 1], self.problem_instance.rho).path_length()) 
            #else:
            #    distance += (compute_length_of_relaxed_dubins_path(states[index - 1], self.problem_instance.sink, self.problem_instance.rho) -
            #                 dubins.shortest_path(tuples[index - 1], tuples[index], self.problem_instance.rho).path_length() -
            #                 compute_length_of_relaxed_dubins_path(states[index], self.problem_instance.sink, self.problem_instance.rho))

            #states.pop(index)
            #tuples.pop(index)
            #individual.pop(index)
        
        return individual, distance

    # TODO: At somepoint this should be converted to use corrolated mutations.
    def angle_mutation (self, individual: CEDOPADSRoute, q: float) -> CEDOPADSRoute:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
        # Core idea here is that in the start we are quite interested in making
        # large jumps in each psi and tau, however in the end we are more interested 
        # in wiggeling them around to perform the last bit of optimization
        for idx, (k, _, _, r) in enumerate(individual):
            u = random.uniform(0, 1)
            if u < q:
                # Compute probabilities of picking an areas of acqusion based on their "arc lengths"
                # with higher arc lengths having having  a higher probability of getting chosen.
                arc_lengths = []
                for aoa in self.problem_instance.nodes[k].AOAs:
                    arc_length = aoa.b - aoa.a if aoa.a < aoa.b else (2 * np.pi - aoa.b) + aoa.a
                    arc_lengths.append(arc_length)

                aoa_probabilities = np.array(arc_lengths) / sum(arc_lengths)
                aoa: AreaOfAcquisition = np.random.choice(self.problem_instance.nodes[k].AOAs, size = 1, p = aoa_probabilities)[0]
                
                # Generate angles from interval
                new_psi = aoa.generate_uniform_angle()
                new_tau = (np.pi + new_psi + (np.random.beta(3, 3) - 0.5) * self.problem_instance.eta)  % (2 * np.pi)
                individual[idx] = (k, new_psi, new_tau, r)

        return individual

    def wiggle_angles(self, individual: CEDOPADSRoute, q: float, scale_modifier: float = 0.02, scale_for_tau: float = 0.3) -> CEDOPADSRoute:
        """Wiggles the angles of the individual slightly, using truncated normal distributions."""
        for idx, (k, psi, _, r) in enumerate(individual):
            if np.random.uniform(0, 1) < q:
                # Figure out which interval the current angle is located within.
                for jdx, aoa in enumerate(self.problem_instance.nodes[k].AOAs):
                    if aoa.contains_angle(psi):
                        break

                # Set bounds for random number generation NOTE: There are checks for these in the data_types.py file.
                # TODO: Consider using beta distribution to compute these
                new_psi = aoa.generate_truncated_normal_angle(mean = self.problem_instance.nodes[k].thetas[jdx], scale_modifier = scale_modifier)
                new_tau = truncnorm.rvs(-self.problem_instance.eta / (2 * scale_for_tau), self.problem_instance.eta / (2 * scale_for_tau), 
                                        loc = np.pi + new_psi, scale = scale_for_tau) % (2 * np.pi)
                individual[idx] = (k, new_psi, new_tau, r)

        return individual

    # ---------------------------------------- Parent Selection ----------------------------------------- #
    def windowing(self) -> ArrayLike:
        """Uses windowing to compute the probabilities that each indvidual in the population is chosen for recombination."""
        minimum_fitness = np.min(self.fitnesses)
        modified_fitnesses = self.fitnesses - minimum_fitness
        return modified_fitnesses / np.sum(modified_fitnesses)

    def sigma_scaling(self, c: float = 2) -> ArrayLike:
        """Uses sigma scaling to compute the probabilities that each indvidual in the population is chosen for recombination."""
        modified_fitnesses = np.maximum(self.fitnesses - (np.mean(self.fitnesses) - c * np.std(self.fitnesses)), 0)
        return modified_fitnesses / np.sum(modified_fitnesses)

    # --------------------------------------- Sampling of parents --------------------------------------- #
    def stochastic_universal_sampling (self, cdf: ArrayLike, m: int) -> List[int]:
        """Implements the SUS algortihm, used to sample the indicies of m parents from the population."""
        indicies = [None for _ in range(m)] 
        i, j = 0, 0
        r = np.random.uniform(0, 1 / m)

        # The core idea behind the algorithm is that instead of doing m-roletewheels to find m parents, 
        # we do a single spin with a roletewheel with m equally spaced indicators (ie. we pick r once and
        # the offset for each of the indicators is 1 / m)
        while i < m:
            while r <= cdf[j]:
                indicies[i] = j 
                r += 1 / m
                i += 1
            
            j += 1

        return indicies

    # ----------------------------------------- Survivor Selection -------------------------------------- #
    def mu_comma_lambda_selection(self, offspring: List[CEDOPADSRoute]) -> List[CEDOPADSRoute]:
        """Picks the mu offspring with the highest fitnesses to populate the the next generation, note this is done with elitism."""
        fitnesses_of_offspring = np.array(list(map(lambda child: self.fitness_function(child), offspring)))

        # Calculate the indicies of the best performing memebers
        indicies_of_new_generation = np.argsort(fitnesses_of_offspring)[-self.mu:]

        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_offspring[indicies_of_new_generation]
        return [offspring[i] for i in indicies_of_new_generation]

    # -------------------------------------------- Main Loop of GA --------------------------------------- #
    def run(self, time_budget: float, progress_bar: bool = False) -> CEDOPADSRoute:
        """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
        start_time = time.time()
        # Generate an initial population, which almost satifies the distant constraint, ie. add nodes until
        # the sink cannot be reached within d - t_max units where d denotes the length of the route.
        self.population = []
        for _ in range(self.mu):
            route = []
            while self.problem_instance.is_route_feasable(route):
                route.append(self.pick_new_visit(route))

            self.population.append(route[:-1])

        # Compute fitnesses of each route in the population and associated probabilities using 
        # the parent_selection_mechanism passed to the method, which sould be one of the parent
        # selection methods defined earlier in the class declaration.
        self.fitnesses = [self.fitness_function(individual) for individual in self.population]

        # NOTE: The fitnesses are set as a field of the class so they are accessable within the method
        probabilities = parent_selection_mechanism() 
        cdf = np.add.accumulate(probabilities)

        gen = 0

        # Also commonly refered to as a super individual
        self.individual_with_highest_recorded_fitness, self.highest_fitness_recorded = None, 0

        with tqdm(total=time_budget, desc="GA", disable = not progress_bar) as pbar:
            # Set initial progress bar information.
            idx_of_individual_with_highest_fitness = np.argmax(self.fitnesses)

            avg_fitness = np.sum(self.fitnesses) / self.mu
            avg_distance = sum(self.problem_instance.compute_length_of_route(route) for route in self.population) / self.mu
            length_of_best_route = self.problem_instance.compute_length_of_route(self.population[idx_of_individual_with_highest_fitness])
            avg_length = sum(len(route) for route in self.population) / self.mu
            pbar.set_postfix({
                    "Gen": gen,
                    "Best": f"({self.fitnesses[idx_of_individual_with_highest_fitness]:.1f}, {length_of_best_route:.1f}, {len(self.population[idx_of_individual_with_highest_fitness])})",
                    "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})"
                })

            while time.time() - start_time < time_budget:
                # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
                # and perform k-point crossover using these parents, to create a total of m^k children which is
                # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
                # the angles in order to find new versions of the parents with higher fitness values.
                
                # TODO: Maybe these can be sorted in asccending order to speed up the average computation time of the SUS algorithm.
                scores = [[self.problem_instance.compute_score_of_visit(visit, self.utility_function) for visit in individual] for individual in self.population]
                # TODO: these states could be generated on the fly, using a dictionary since not all individuals in the population is chosen 
                states = [self.problem_instance.get_states(individual) for individual in self.population]

                offspring = []
                while len(offspring) < self.lmbda:
                    parent_indicies = self.stochastic_universal_sampling(cdf, m = m)
                    for child in crossover_mechanism([self.population[i] for i in parent_indicies], [scores[i] for i in parent_indicies], [states[i] for i in parent_indicies]): 
                        mutated_child = self.mutate(deepcopy(child), p_s = 0.1, p_i = 0.3, p_r = 0.2, q = 0.1)
                        fixed_mutated_child, _ = self.fix_length(mutated_child) # NOTE: This also returns the length of the mutated_child, which could be usefull in later stages.
                        offspring.append(fixed_mutated_child)

                        # TODO: implement a local search improvement operator, ie. convert
                        #       the algorithm to a Lamarckian memetatic algortihm

                    # NOTE: we are simply interested in optimizing the existing angles within the individual,
                    # hence we create a new offspring where a few of the angles are different compared to 
                    # the original individual, sort of like a Lamarckian mematic algortihm, however
                    # here the local search is done using a stochastic operator.
                    #offspring.append(self.fix_length(self.wiggle_angles(deepcopy(self.population[i]), q = 0.05)))
                    #index_of_individual_to_be_wiggled = parent_indicies[np.random.choice(len(parent_indicies))]
                    #offspring.append(self.fix_length(self.wiggle_angles(deepcopy(self.population[index_of_individual_to_be_wiggled]), q = 0.2))[0])

                # Survivor selection, responsible for find the lements which are passed onto the next generation.
                gen += 1
                self.population = survivor_selection_mechanism(offspring)
                probabilities = parent_selection_mechanism() 
                cdf = np.add.accumulate(probabilities)

                idx_of_individual_with_highest_fitness = np.argmax(self.fitnesses)
                if self.fitnesses[idx_of_individual_with_highest_fitness] > self.highest_fitness_recorded:
                    self.individual_with_highest_recorded_fitness = self.population[idx_of_individual_with_highest_fitness]
                    self.highest_fitness_recorded = self.fitnesses[idx_of_individual_with_highest_fitness]
                    self.length_of_individual_with_highest_recorded_fitness = self.problem_instance.compute_length_of_route(self.individual_with_highest_recorded_fitness)

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                if displa
                pbar.n = round(time.time() - start_time, 1)
                avg_fitness = np.mean(self.fitnesses)
                avg_distance = sum(self.problem_instance.compute_length_of_route(route) for route in self.population) / self.mu
                length_of_best_route = self.problem_instance.compute_length_of_route(self.population[idx_of_individual_with_highest_fitness])
                avg_length = sum(len(route) for route in self.population) / self.mu
                pbar.set_postfix({
                        "Gen": gen,
                        "Best": f"({self.highest_fitness_recorded:.1f}, {self.length_of_individual_with_highest_recorded_fitness:.1f}, {len(self.individual_with_highest_recorded_fitness)})",
                        "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})"
                })

        return self.individual_with_highest_recorded_fitness

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.f.d.b.txt", needs_plotting = True)
    mu = 128
    ga = GA(problem_instance, utility_function, mu = mu, lmbda = mu * 7)
    #import cProfile
    #cProfile.run("ga.run(300, 3, ga.sigma_scaling, ga.unidirectional_greedy_crossover, ga.mu_comma_lambda_selection, progress_bar = True)", sort = "cumtime")
    route = add_free_visits(ga.run(300, 3, ga.sigma_scaling, ga.unidirectional_greedy_crossover, ga.mu_comma_lambda_selection, progress_bar = True))
    problem_instance.plot_with_route(route, utility_function)
    plt.show()
