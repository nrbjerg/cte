# %%
import dubins
from copy import deepcopy
import random
import time
from typing import List, Optional, Callable, Set, Tuple
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle,AreaOfAcquisition, Matrix, Position, Vector
from numpy.typing import ArrayLike
import hashlib
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from library.CEDOPADS_solvers.local_search_operators import remove_visit, greedily_add_visits_while_posible, add_free_visits, get_equdistant_samples
from dataclasses import dataclass, field

@dataclass()
class Individual:
    """Stores the information relating to an individual"""
    route: CEDOPADSRoute
    mutation_step_sizes: Tuple[Vector] 

    # Optional fields 
    scores: List[float] = None
    states: List[State] = None 
    segment_lengths: List[float] = None 

    def __len__(self) -> int:
        """Returns the length of the route"""
        return len(self.route)

def compute_sdr_tensor(positions: Matrix, scores: ArrayLike, c_s: float, c_d: float) -> ArrayLike:
    """Computes a tensor consisting of all of the distances between the nodes"""
    n = positions.shape[0] - 2
    tensor = np.zeros((n + 2, n + 2, n))
    for i in range(n + 2):
        for j in range(n + 2):
            for k in range(n):
                if i == j:
                    continue
                #if i == k or j == k:
                #    continue

                # Compute the length between the line segment between position i and position j and the point at position k
                # link to an explaination of the formula: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                area = np.abs((positions[i][1] - positions[j][1]) * positions[k][0] - (positions[i][0] - positions[j][0]) * positions[k][1] + positions[i][0] * positions[j][1] - positions[i][1] * positions[j][0])

                distance = (area + 0.01) / (np.linalg.norm(positions[i] - positions[j]) + 0.01) # We need to not divide by 0 both here and later.
                # NOTE: the 0.01 is used in case the area is 0, which happends if the points at positions i,j and k are colinear
                tensor[i, j, k] = scores[k] ** c_s / distance ** c_d # NOTE: The 0.01 is used to prevent nummerical errors, if we get a 0 we might divide by inf.
                tensor[j, i, k] = tensor[i, j, k]

    return tensor

class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem_instance: CEDOPADSInstance
    utility_function: UtilityFunction
    mu: int # Population Number
    lmbda: int # Number of offspring
    number_of_nodes: int
    node_indices: Set[int]
    node_indices_array: Vector

    def __init__ (self, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, mu: int = 256, lmbda: int = 256 * 7, xi: int = 0, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        self.problem_instance = problem_instance
        self.utility_function = utility_function
        self.lmbda = lmbda
        self.mu = mu
        self.xi = xi
        self.number_of_nodes = len(self.problem_instance.nodes)
        self.node_indices = set(range(self.number_of_nodes))
        self.node_indices_arrray = np.array(range(self.number_of_nodes))
        
        # Seed RNG for repeatability
        if seed is None:
            hash_of_id = hashlib.sha1(problem_instance.problem_id.encode("utf-8")).hexdigest()[:8]
            seed = int(hash_of_id, 16)
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Make sure that the sink can be reached from the source otherwise there is no point in running the algorithm.
        if (d := np.linalg.norm(self.problem_instance.source - self.problem_instance.sink)) > self.problem_instance.t_max:
            raise ValueError(f"Cannot construct a valid dubins route since ||source - sink|| = {d} is greater than t_max = {self.problem_instance.t_max}")

    # ------------------------------------ Initialization -------------------------------------------- # 
    def initialize_population (self) -> List[CEDOPADSRoute]:
        """Initializes a random population of routes, for use in the genetic algorithm."""
        population = []
        for _ in range(self.mu):
            route = [] 
            blacklist = set()

            while self.problem_instance.is_route_feasable(route):
                # Pick a new random visit and add it to the route.
                k = random.choice(list(self.node_indices.difference(blacklist)))
                blacklist.add(k)
                psi = random.choice(self.problem_instance.nodes[k].AOAs).generate_uniform_angle() # TODO:
                tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
                r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

                route.append((k, psi, tau, r)) 

            # Initialize what we need to know from the individual
            scores = self.problem_instance.compute_scores_along_route(route, self.utility_function)
            population.append(Individual(route, scores = scores, mutation_step_sizes = scores)) # TODO:

        return population

    # ---------------------------------------- Fitness function ------------------------------------------ 
    def fitness_function(self, individual: Individual) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        return sum(individual.scores)
     
    # -------------------------------------- Recombination Operators -------------------------------------- #
    def partially_mapped_crossover(self, parents: List[Individual]) -> List[Individual]:
        """Performs a partially mapped crossover (PMX) operation on the parents to generate two offspring"""
        m = min(len(parent) for parent in parents)
        if not m >= 2:
            return parents 

        indices = sorted(np.random.choice(m, 2, replace = False))

        offspring = []
        for idx in range(2):
            paternal, maternal = parents[idx], parents[(idx + 1) % 2] 
            
            # TODO: Needs to be updated to include step sizes
            child_route = paternal.route[:indices[0]] + maternal.route[indices[0]:indices[1]] + paternal.route[indices[1]:]

            # Fix any mishaps when the child was created, i.e. make sure that the same node i never visited twice.
            ks = {child_route[jdx][0] for jdx in range(indices[0], indices[1] + 1)} 

            # NOTE: Normaly the order of the visits is reversed, however since our distance "metric" is in this case non-symetric we will not use this convention.
            visits_which_may_replace = [visit for visit in paternal.route[indices[0]:indices[1]] if visit[0] not in ks]
            number_of_visits_replaced, number_of_visits_which_can_be_replaced = 0, len(visits_which_may_replace)

            indices_to_remove = []
            for jdx in itertools.chain(range(indices[0]), range(indices[1], len(child_route))):
                # Replace the visit with one from the other parent is posible otherwise simply remove the visit entirely
                if child_route[jdx][0] in ks:
                    if number_of_visits_replaced < number_of_visits_which_can_be_replaced:
                        child_route[jdx] = visits_which_may_replace[number_of_visits_replaced]
                        number_of_visits_replaced += 1
                    else:
                        indices_to_remove.append(jdx)

            # actually remove the indices, we have to do it to not mess with the indices.
            for jdx in reversed(indices_to_remove):
                child_route.pop(jdx)

            offspring.append(Individual(child_route, mutation_step_sizes = paternal.mutation_step_sizes))

        return offspring
    
    #--------------------------------------------- Mutation Operators -------------------------------------- 
    def mutate(self, individual: Individual, p: float, q: float, eps: float = 0.01) -> Individual: 
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route.""" 
        # Replace visits within route
        n = max(0, min(np.random.geometric(1 - p) - 1, len(individual) - 1))
        indices = sorted(np.random.choice(len(individual), n, replace = False))
        for i in indices:
            # Replace visits through route.
            route_without_visit_i = individual.route[:i] + individual.route[i + 1:]
            best_visit = self.pick_visit_to_insert_based_on_SDR(route_without_visit_i, i)

            # Actually replace the visit in the individual
            individual.route[i] = best_visit
            individual.mutation_step_sizes[i] = self.problem_instance.nodes[best_visit[0]].get_AOA(best_visit[1]).phi / 8

        # Insert visists within the route the route.
        n = max(0, min(np.random.geometric(1 - q) - 1, len(individual)))
        indices = sorted(np.random.choice(len(individual) + 1, n, replace = False))
        for i in reversed(indices):
            best_visit = self.pick_visit_to_insert_based_on_SDR(individual.route, i)

            # Actually insert the information into the individual
            individual.route.insert(i, best_visit)
            individual.mutation_step_sizes.insert(i,  self.problem_instance.nodes[best_visit[0]].get_AOA(best_visit[1]).phi / 8)

        # Get the new indices of insertion, to make sure that these are not removed by the fix_length operator

        # Finally we perform multiple angle mutations
        return self.mutate_angles(individual, eps)

    def pick_visit_to_insert_based_on_SDR(self, route: CEDOPADSRoute, i: int) -> Visit:
        """Picks a new visit to insert based on the SDR scores of each visit of a randomly picked node, which is picked according to the SDR tensor"""
        if i == 0:
                # We need to find a visit to insert at the first position in the route.
            self.sdrs_for_overwrite[:] = self.sdr_tensor[route[0][0]][self.number_of_nodes][:]

        elif i == len(route): # NOTE: since we are going from high to low, this is fine.
                # We need to find a visit to insert at the first position in the route.
            self.sdrs_for_overwrite[:] = self.sdr_tensor[route[-1][0]][self.number_of_nodes + 1][:]

        else:
                # We need to find a node to insert in between visits i - 1 and i
            self.sdrs_for_overwrite[:] = self.sdr_tensor[route[i - 1][0]][route[i][0]][:]
                
        self.mask_for_overwrite[:] = self.default_mask[:]

        for (k, _, _, _) in route:
            self.mask_for_overwrite[k] = False

        weights = self.sdrs_for_overwrite[self.mask_for_overwrite]
        probs = weights / np.sum(weights)

        # TODO: use the actual path where the visit is inserted to determine the appropriate k
        k = np.random.choice(self.node_indices_arrray[self.mask_for_overwrite], p = probs)

        # Find the actual best visit for visiting node k according to the SDR score.
        best_visit = None
        sdr_of_best_visit = 0 

        if i == 0:
            q = self.problem_instance.get_state(route[0])
            original_distance = compute_length_of_relaxed_dubins_path(q.angle_complement(), self.problem_instance.source, self.problem_instance.rho)
            for visit in get_equdistant_samples(problem_instance, k):
                q_v = self.problem_instance.get_state(visit) 
                score_of_visit = self.problem_instance.compute_score_of_visit(visit, self.utility_function)

                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_v.angle_complement(), self.problem_instance.source, self.problem_instance.rho) +  dubins.shortest_path(q_v.to_tuple(), q.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    best_visit = visit 
                    break

                elif (sdr_of_visit := score_of_visit / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit

        elif i == len(route):
            q = self.problem_instance.get_state(route[i - 1])
            original_distance = compute_length_of_relaxed_dubins_path(q, self.problem_instance.sink, self.problem_instance.rho)
            for visit in get_equdistant_samples(problem_instance, k):
                q_v = self.problem_instance.get_state(visit) 
                score_of_visit = self.problem_instance.compute_score_of_visit(visit, self.utility_function)

                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_v, self.problem_instance.sink, self.problem_instance.rho) +  dubins.shortest_path(q.to_tuple(), q_v.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    best_visit = visit 
                    break

                elif (sdr_of_visit := score_of_visit / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit


        else: 
            q_i, q_t = self.problem_instance.get_state(route[i - 1]), self.problem_instance.get_state(route[i])
            original_distance = dubins.shortest_path(q_i.to_tuple(), q_t.to_tuple(), self.problem_instance.rho).path_length()
            for visit in get_equdistant_samples(problem_instance, k):
                q_v = self.problem_instance.get_state(visit) 
                score_of_visit = self.problem_instance.compute_score_of_visit(visit, self.utility_function)

                distance_through_visit = (dubins.shortest_path(q_i.to_tuple(), q_v.to_tuple(), self.problem_instance.rho).path_length() + 
                                              dubins.shortest_path(q_v.to_tuple(), q_t.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    best_visit = visit 
                    break

                elif (sdr_of_visit := score_of_visit / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit

        return best_visit

    # TODO: At somepoint this should be converted to use corrolated mutations.
    def mutate_angles (self, individual: Individual, eps: float = 0.01) -> Individual:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
        # 1. Mutate 
        scale = 0.05
        # Core idea here is that in the start we are quite interested in making
        # large jumps in each psi and tau, however in the end we are more interested 
        # in wiggeling them around to perform the last bit of optimization
        n = max(0, min(np.random.geometric(1 - 0.1) - 1, len(individual)))
        for idx in np.random.choice(len(individual), n, replace = False):
            (k, psi, _, r) = individual.route[idx]
            if np.random.uniform(0, 1) < 0.9: # Stay within the same AOA but wiggle the values sligtly.
                # Compute probabilities of picking an areas of acqusion based on their "arc lengths"
                # with higher arc lengths having having  a higher probability of getting chosen.
                aoa = self.problem_instance.nodes[k].get_AOA(psi)

                # TODO: Consider using beta distribution to compute these
                if aoa.a < aoa.b:
                    new_psi = truncnorm.rvs((aoa.a - psi) / scale, (aoa.b - psi)  / scale, loc = psi, scale = scale)
                else:
                    new_psi = truncnorm.rvs((aoa.a - psi)  / scale, (aoa.b + 2 * np.pi - psi) / scale, loc = psi, scale = scale) % (2 * np.pi)

                new_tau = (new_psi + np.pi + truncnorm.rvs((-self.problem_instance.eta / 2) / scale, (self.problem_instance.eta / 2) / scale,
                                        loc = 0, scale = scale)) % (2 * np.pi)

                new_r = truncnorm.rvs((self.problem_instance.sensing_radii[0] - r) / scale, (self.problem_instance.sensing_radii[1] - r) / scale, loc = r, scale = scale)
                individual.route[idx] = (k, new_psi, new_tau, new_r)

            else: # Pick a random new angle uniformly.
                arc_lengths = []
                for aoa in self.problem_instance.nodes[k].AOAs:
                    arc_length = aoa.b - aoa.a if aoa.a < aoa.b else (2 * np.pi - aoa.b) + aoa.a
                    arc_lengths.append(arc_length)

                aoa_probabilities = np.array(arc_lengths) / sum(arc_lengths)
                aoa: AreaOfAcquisition = np.random.choice(self.problem_instance.nodes[k].AOAs, size = 1, p = aoa_probabilities)[0]

                # Generate new visit randomly.
                new_psi = aoa.generate_uniform_angle()
                new_tau = ((np.pi + new_psi) + (np.random.beta(4, 4) - 0.5) * self.problem_instance.eta)  % (2 * np.pi)
                new_r = np.random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])
                individual.route[idx] = (k, new_psi, new_tau, new_r)

        return individual

    # --------------------------------------------------------- Feasability ------------------------------------------------------------------- #
    # NOTE: Mayor performance gains where probabily found here, but the code needs to be tested.
    def fix_length_non_optimized(self, individual: Individual, indices_to_skip: List[int] = None) -> Individual:
        """Removes visits from the individual until it obeys the distance constraint."""
        # Keep track of these rather than recomputing them on the fly
        individual.scores = self.problem_instance.compute_scores_along_route(individual.route, self.utility_function)
        individual.states = self.problem_instance.get_states(individual.route)
        individual.segment_lengths = self.problem_instance.compute_lengths_of_route_segments_from_states(individual.states)

        distance = sum(individual.segment_lengths) 

        if distance > self.problem_instance.t_max:
            # Compute sdsr "score distance saved ratio" of each visit these will be updated throughout

            idx_skipped = None
            while distance > self.problem_instance.t_max: 
                # Iteratively remove visits until the constraint is no longer violated, and update the distance saved throughout.
                # lengths_of_new_segments[i] will corespond to the new length of the ith path segment, if visit number i is skipped
                lengths_of_new_segments = [compute_length_of_relaxed_dubins_path(individual.states[1].angle_complement(), self.problem_instance.source, self.problem_instance.rho)]

                # distances_saved[i] is the distance saved by skipping the visit at index i.
                distances_saved = [(individual.segment_lengths[0] + individual.segment_lengths[1]) - lengths_of_new_segments[0]]

                # SDSR = (score distance saved ratio)
                sdsr_scores = [individual.scores[0] / distances_saved[0]]

                for idx in range(1, len(individual) - 1):
                    lengths_of_new_segments.append(dubins.shortest_path(individual.states[idx - 1].to_tuple(), individual.states[idx + 1].to_tuple(), self.problem_instance.rho).path_length()) 
                    distances_saved.append((individual.segment_lengths[idx] + individual.segment_lengths[idx + 1]) - lengths_of_new_segments[idx])
                    sdsr_scores.append(individual.scores[idx] / distances_saved[idx])

                lengths_of_new_segments.append(compute_length_of_relaxed_dubins_path(individual.states[-2], self.problem_instance.sink, self.problem_instance.rho))
                distances_saved.append((individual.segment_lengths[-2] + individual.segment_lengths[-1]) - lengths_of_new_segments[-1])
                sdsr_scores.append(individual.scores[-1] / distances_saved[-1])

                # This is according to the ratio: score of visit / length saved by excluding visit that is the "score distance saved ratio (SDSR)"
                idx_skipped = np.argmin(sdsr_scores)

                individual.route.pop(idx_skipped)
                individual.states.pop(idx_skipped)
                individual.scores.pop(idx_skipped)
                individual.psi_mutation_step_sizes.pop(idx_skipped)
                individual.tau_mutation_step_sizes.pop(idx_skipped)
            
                # NOTE: we need to change both the segment length at the idx_skipped and idx_skipped + 1, since we are skipping the 
                #       visit at idx_skipped and this visit is partially responsible for these path segements
                individual.segment_lengths[idx_skipped + 1] = lengths_of_new_segments[idx_skipped]
                individual.segment_lengths.pop(idx_skipped)

                distance -= distances_saved[idx_skipped]

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
    def mu_comma_lambda_selection(self, offspring: List[CEDOPADSRoute]) -> List[CEDOPADSRoute]:
        """Picks the mu offspring with the highest fitnesses to populate the the next generation, note this is done with elitism."""
        fitnesses_of_offspring = np.array(list(map(lambda child: self.fitness_function(child), offspring)))

        # Calculate the indices of the best performing memebers
        indices_of_new_generation = np.argsort(fitnesses_of_offspring)[-self.mu:]

        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_offspring[indices_of_new_generation]
        return [offspring[i] for i in indices_of_new_generation]

    # -------------------------------------------- Main Loop of GA --------------------------------------- #
    def run(self, time_budget: float, display_progress_bar: bool = False) -> CEDOPADSRoute:
        """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
        start_time = time.time()
        # Generate an initial population, which almost satifies the distant constraint, ie. add nodes until
        # the sink cannot be reached within d - t_max units where d denotes the length of the route.
        self.population = self.initialize_population()

        # Compute distances which will be used to guide the mutation operator
        positions = np.concat((np.array([node.pos for node in self.problem_instance.nodes]), self.problem_instance.source.reshape(1, 2), self.problem_instance.sink.reshape(1, 2)))
        self.sdr_tensor = compute_sdr_tensor(positions, np.array([node.base_line_score for node in self.problem_instance.nodes]), c_s = 2, c_d = 2)
        self.sdrs_for_overwrite = np.empty_like(self.sdr_tensor[0][0]) 
        self.default_mask = np.ones(self.number_of_nodes, dtype=bool)
        self.mask_for_overwrite = np.empty_like(self.default_mask)

        # Compute fitnesses of each route in the population and associated probabilities using 
        # the parent_selection_mechanism passed to the method, which sould be one of the parent
        # selection methods defined earlier in the class declaration.
        self.fitnesses = [self.fitness_function(individual) for individual in self.population]

        # NOTE: The fitnesses are set as a field of the class so they are accessable within the method
        probabilities = self.sigma_scaling()
        cdf = np.add.accumulate(probabilities)

        gen = 0

        # Also commonly refered to as a super individual
        self.individual_with_highest_recorded_fitness, self.highest_fitness_recorded = None, 0

        with tqdm(total=time_budget, desc="GA", disable = not display_progress_bar) as pbar:
            # Set initial progress bar information.
            idx_of_individual_with_highest_fitness = np.argmax(self.fitnesses)

            if display_progress_bar:
                avg_fitness = np.sum(self.fitnesses) / self.mu
                avg_distance = sum(self.problem_instance.compute_length_of_route(individual.route) for individual in self.population) / self.mu
                length_of_best_route = self.problem_instance.compute_length_of_route(self.population[idx_of_individual_with_highest_fitness].route)
                avg_length = sum(len(individual) for individual in self.population) / self.mu
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
                
                offspring = []
                while len(offspring) < self.lmbda - self.xi: 
                    parent_indices = self.stochastic_universal_sampling(cdf, m = 2)
                    parents = [self.population[i] for i in parent_indices]
                    for child in self.partially_mapped_crossover(parents): 
                        mutated_child = self.mutate(deepcopy(child), p = 0.2, q = 0.3) # NOTE: we need this deepcopy!
                        fixed_mutated_child = self.fix_length(mutated_child) # NOTE: This also could return the length of the mutated_child, which could be usefull in later stages.
                        offspring.append(fixed_mutated_child)
                
                parents_to_be_mutated = [self.population[i] for i in self.stochastic_universal_sampling(cdf, self.xi)]
                for parent in parents_to_be_mutated:
                    mutated_parent = self.mutate_angles(parent) 
                    fixed_mutated_parent = self.fix_length(mutated_parent) 
                    offspring.append(fixed_mutated_parent)

                    # TODO: Test if it is adviceable to pick a small subset of the population to evolve along with the new

                # Survivor selection, responsible for find the lements which are passed onto the next generation.
                gen += 1
                self.population = self.mu_comma_lambda_selection(offspring)
                probabilities = self.sigma_scaling() 
                cdf = np.add.accumulate(probabilities)

                idx_of_individual_with_highest_fitness = np.argmax(self.fitnesses)
                if self.fitnesses[idx_of_individual_with_highest_fitness] > self.highest_fitness_recorded:
                    self.individual_with_highest_recorded_fitness = self.population[idx_of_individual_with_highest_fitness]
                    self.highest_fitness_recorded = self.fitnesses[idx_of_individual_with_highest_fitness]
                    self.length_of_individual_with_highest_recorded_fitness = self.problem_instance.compute_length_of_route(self.individual_with_highest_recorded_fitness.route)

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                if display_progress_bar:
                    pbar.n = round(time.time() - start_time, 1)
                    avg_fitness = np.mean(self.fitnesses)
                    avg_distance = sum(sum(individual.segment_lengths) for individual in self.population) / self.mu
                    length_of_best_route = sum(self.individual_with_highest_recorded_fitness.segment_lengths) 
                    avg_length = sum(len(route) for route in self.population) / self.mu
                    pbar.set_postfix({
                            "Gen": gen,
                            "Best": f"({self.highest_fitness_recorded:.1f}, {self.length_of_individual_with_highest_recorded_fitness:.1f}, {len(self.individual_with_highest_recorded_fitness)})",
                            "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})",
                            "Var": f"{np.var(self.fitnesses):.1f}"
                    })

        return self.individual_with_highest_recorded_fitness.route

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.f.d.b.txt", needs_plotting = True)
    mu = 128
    ga = GA(problem_instance, utility_function, mu = mu, lmbda = mu * 7, xi = int(np.floor(mu / 4)))
    #import cProfile
    #cProfile.run("ga.run(300, display_progress_bar = True)", sort = "cumtime")
    route = ga.run(300, display_progress_bar=True)
    augmented_route = add_free_visits(problem_instance, route, utility_function)
    print(f"Score of augmented route: {problem_instance.compute_score_of_route(augmented_route, utility_function)}")
    problem_instance.plot_with_route(route, utility_function)
    plt.show()

# %%
