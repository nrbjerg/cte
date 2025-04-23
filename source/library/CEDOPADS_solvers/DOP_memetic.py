# %%
import dubins
import multiprocessing
from copy import deepcopy
from itertools import product
import random
import time
from typing import List, Optional, Callable, Set, Tuple
import numpy as np
from dataclasses import dataclass, field
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical, utility_baseline
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle,AreaOfAcquisition, Matrix, Position, Vector
from numpy.typing import ArrayLike
import hashlib
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import truncnorm
from library.CEDOPADS_solvers.local_search_operators import remove_visit, greedily_add_visits_while_posible, add_free_visits, get_equdistant_samples

@njit()
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

@dataclass 
class Indiviudal:
    """Models an individual with a mutation step size vector."""
    route: CEDOPADSRoute
    mutation_step_sizes: List[float]
    scores: List[float] = field(init = False)
    states: List[State] = field(init = False)
    segment_lengths: List[float] = field(init = False)

    def __len__ (self) -> int:
        return len(self.route)

class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem_instance: CEDOPADSInstance
    utility_function: UtilityFunction
    mu: int # Population Number
    lmbda: int # Number of offspring
    xi: int # Number of individuals which goes directly to next generation
    number_of_nodes: int
    node_indicies: Set[int]
    node_indicies_array: Vector
    fitness_idx: int = field(init = False)

    def __init__ (self, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, mu: int = 256, lmbda: int = 256 * 7, xi: int = 128, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        self.problem_instance = problem_instance
        self.utility_function = utility_function
        self.lmbda = lmbda
        self.mu = mu
        self.xi = xi
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

    # ------------------------------------ Initialization -------------------------------------------- # 
    def pick_new_visit(self, individual: CEDOPADSRoute) -> Visit:
        """Picks a new place to visit, according to how much time is left to run the algorithm."""
        # For the time being simply pick a random node to visit FIXME: Should be updated, 
        # maybe we pick them with a probability which is propotional to the distance to the individual?
        blacklist = set(k for (k, _, _, _) in individual)

        k = random.choice(list(self.node_indicies.difference(blacklist)))
        tau = np.random.uniform(0, 2 * np.pi) 
        #tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)

        return (k, 0, tau, 0)

    def initialize_population (self) -> List[CEDOPADSRoute]:
        """Initializes a random population of routes, for use in the genetic algorithm."""
        population = []
        while len(population) < self.mu:
            route = []
            while self.problem_instance.is_route_feasable(route):
                route.append(self.pick_new_visit(route))

            if len(route) != 1:
                population.append(Indiviudal(route[:-1], [0.5 for _ in range(len(route) - 1)]))

        return population

    # -------------------------- General functions for mutation operators ects ------------------------------ #
    def pick_new_visit_based_on_sdr_and_index(self, individual: Indiviudal, idx: int, m: int = 5) -> Visit:
        """Picks a new visit based on the ''neighbours'' of the visit as well as the euclidian distance between
           the line segment between the nodes, and the new node, that is the candidate to be added."""
        if idx == 0:
            i = individual.route[0][0], 
            j = self.number_of_nodes
            q_idx = self.problem_instance.get_state(individual[idx])
            original_distance = compute_length_of_relaxed_dubins_path(q_idx.angle_complement(), self.problem_instance.source, self.problem_instance.rho)
        elif idx == len(individual) - 1:
            i = individual.route[-1][0]
            j = self.number_of_nodes + 1
            q_idx = self.problem_instance.get_state(individual[idx])
            original_distance = compute_length_of_relaxed_dubins_path(q_idx, self.problem_instance.sink, self.problem_instance.rho)
        else:
            i = individual.route[idx][0]
            j = individual.route[idx + 1][0]
            q_idx = self.problem_instance.get_state(individual[idx])
            original_distance = dubins.shortest_path(q_idx.to_tuple(), self.problem_instance.get_state(individual[idx + 1]).to_tuple(), self.problem_instance.rho).path_length()

        self.sdrs_for_overwrite[:] = self.sdr_tensor[i][j][:]
        self.mask_for_overwrite[:] = self.default_mask[:]

        for (k, _, _, _) in individual.route:
            self.mask_for_overwrite[k] = False

        weights = self.sdrs_for_overwrite[self.mask_for_overwrite]
        probs = weights / np.sum(weights)
        k = np.random.choice(self.node_indicies_arrray[self.mask_for_overwrite], p = probs)

        # perform sampling and pick the best one from a set of m posible.
        visits_and_sdrs = []
        for visit in [(k, 0, n * np.pi / 3, 0) for n in range(6)]:
            q_v = self.problem_instance.get_state(visit)

            # NOTE This may not be needed if the score of the visits generated by the sampling does not vary.
            score_of_visit = self.problem_instance.compute_score_of_visit(visit, self.utility_function)

            if idx == 0:
                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_v.angle_complement(), self.problem_instance.source, self.problem_instance.rho) + 
                                          dubins.shortest_path(q_v.to_tuple(), q_idx.to_tuple(), self.problem_instance.rho).path_length())
            elif idx == len(individual) - 1:
                 distance_through_visit = (dubins.shortest_path(q_idx.to_tuple(), q_v.to_tuple(), self.problem_instance.rho).path_length() + 
                                           compute_length_of_relaxed_dubins_path(q_v, self.problem_instance.sink, self.problem_instance.rho))
            else:
                distance_through_visit = (dubins.shortest_path(q_idx.to_tuple(), q_v.to_tuple(), self.problem_instance.rho).path_length() + 
                                          dubins.shortest_path(q_v.to_tuple(), self.problem_instance.get_state(individual[idx + 1]).to_tuple(), self.problem_instance.rho).path_length())
            
            if distance_through_visit == original_distance:
                return visit

            sdr_of_visit = score_of_visit / (distance_through_visit - original_distance)
            visits_and_sdrs.append((visit, sdr_of_visit))

        return random.choice(sorted(visits_and_sdrs, key = lambda pair: pair[1])[:m])[0] 
                
    # ---------------------------------------- Fitness function ------------------------------------------ 
    def fitness_function(self, individual: Indiviudal) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        score = self.problem_instance.compute_score_of_route(individual.route, self.utility_function) 
        if self.fitness_idx == 0:
            return score
        else:
            length = self.problem_instance.compute_length_of_route(individual.route)
            return (score ** self.fitness_idx) / np.sqrt(0.01 + length)
     
    # -------------------------------------- Recombination Operators --------------------------------------  
    def partially_mapped_crossover(self, parents: List[Indiviudal]) -> List[Indiviudal]:
        """Performs a partially mapped crossover (PMX) operation on the parents to generate two offspring"""
        m = min(len(parent) for parent in parents)
        if not m >= 2:
            return parents 

        indicies = sorted(np.random.choice(m, 2, replace = False))

        offspring = []
        for idx in range(2):
            paternal, maternal = parents[idx], parents[(idx + 1) % 2] 
            
            # TODO: Needs to be updated to include step sizes
            route = paternal.route[:indicies[0]] + maternal.route[indicies[0]:indicies[1]] + paternal.route[indicies[1]:]
            mutation_step_sizes = paternal.mutation_step_sizes[:indicies[0]] + maternal.mutation_step_sizes[indicies[0]:indicies[1]] + paternal.mutation_step_sizes[indicies[1]:]

            # Fix any mishaps when the child was created, i.e. make sure that the same node i never visited twice.
            ks = {route[jdx][0] for jdx in range(indicies[0], indicies[1] + 1)} 

            # NOTE: Normaly the order of the visits is reversed, however since our distance "metric" is in this case non-symetric we will not use this convention.
            indicies_which_may_replace = [i for i in range(indicies[0], indicies[1]) if paternal.route[i][0] not in ks]
            number_of_visits_replaced, number_of_visits_which_can_be_replaced = 0, len(indicies_which_may_replace)

            indicies_to_remove = []
            for jdx in itertools.chain(range(indicies[0]), range(indicies[1], len(route))):
                # Replace the visit with one from the other parent if posible otherwise simply remove the visit entirely
                if route[jdx][0] in ks:
                    if number_of_visits_replaced < number_of_visits_which_can_be_replaced:
                        route[jdx] = paternal.route[indicies_which_may_replace[number_of_visits_replaced]]
                        mutation_step_sizes[jdx] = paternal.mutation_step_sizes[indicies_which_may_replace[number_of_visits_replaced]]
                        number_of_visits_replaced += 1
                    else:
                        indicies_to_remove.append(jdx)

            # actually remove the indices, we have to do it to not mess with the indicies.
            for jdx in reversed(indicies_to_remove):
                route.pop(jdx)
                mutation_step_sizes.pop(jdx)

            offspring.append(Indiviudal(route, mutation_step_sizes)) 

        return offspring
  
    #--------------------------------------------- Mutation Operators -------------------------------------- 
    def pick_visit_to_insert_based_on_SDR(self, route: CEDOPADSRoute, i: int) -> Visit:
        """Picks a new visit to insert based on the SDR scores of each visit of a randomly picked node, which is picked according to the SDR tensor"""
        # Set SDRs for overwrite, remembering that the index self.number_of_nodes corresponds to the source and index self.number_of_nodes + 1 coresponds to the sink
        if i == 0:
            self.sdrs_for_overwrite[:] = self.sdr_tensor[route[0][0]][self.number_of_nodes][:]
        elif i == len(route): # NOTE: since we are going from high to low, this is fine.
            self.sdrs_for_overwrite[:] = self.sdr_tensor[route[-1][0]][self.number_of_nodes + 1][:]
        else:
            self.sdrs_for_overwrite[:] = self.sdr_tensor[route[i - 1][0]][route[i][0]][:]
                
        self.mask_for_overwrite[:] = self.default_mask[:]

        for (k, _, _, _) in route:
            self.mask_for_overwrite[k] = False

        weights = self.sdrs_for_overwrite[self.mask_for_overwrite]
        probs = weights / np.sum(weights)

        # TODO: use the actual path where the visit is inserted to determine the appropriate k
        k = np.random.choice(self.node_indicies_arrray[self.mask_for_overwrite], p = probs)

        # Find the actual best visit for visiting node k according to the SDR score.
        best_visit = None
        sdr_of_best_visit = 0 

        if i == 0:
            # Compute the original distance before including the visit
            q = self.problem_instance.get_state(route[0])
            original_distance = compute_length_of_relaxed_dubins_path(q.angle_complement(), self.problem_instance.source, self.problem_instance.rho)
            for visit in [(k, 0, n * np.pi / 3, 0) for n in range(6)]:
                # Compute score associated with visit and the distance through the visit
                q_v = self.problem_instance.get_state(visit) 
                score_of_visit = self.problem_instance.compute_score_of_visit(visit, self.utility_function)
                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_v.angle_complement(), self.problem_instance.source, self.problem_instance.rho) + 
                                          dubins.shortest_path(q_v.to_tuple(), q.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    return visit 

                elif (sdr_of_visit := score_of_visit / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit

        elif i == len(route):
            # Compute the original distance before including the visit
            q = self.problem_instance.get_state(route[-1])
            original_distance = compute_length_of_relaxed_dubins_path(q, self.problem_instance.sink, self.problem_instance.rho)
            for visit in [(k, 0, n * np.pi / 3, 0) for n in range(6)]:
                # Compute score associated with visit and the distance through the visit
                q_v = self.problem_instance.get_state(visit) 
                score_of_visit = self.problem_instance.compute_score_of_visit(visit, self.utility_function)
                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_v, self.problem_instance.sink, self.problem_instance.rho) +
                                          dubins.shortest_path(q.to_tuple(), q_v.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    return visit 

                elif (sdr_of_visit := score_of_visit / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit

        else: 
            # Compute the original distance before including the visit
            q_i, q_t = self.problem_instance.get_state(route[i - 1]), self.problem_instance.get_state(route[i])
            original_distance = dubins.shortest_path(q_i.to_tuple(), q_t.to_tuple(), self.problem_instance.rho).path_length()
            for visit in [(k, 0, n * np.pi / 3, 0) for n in range(6)]:
                # Compute score associated with visit and the distance through the visit
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


    # TODO: we could look at the pheno type which corresponds to the individdual (i.e. what does the actual route look like.)
    #       when inserting new visits into the routes, this could speed up the convergence rate.
    def mutate(self, individual: Indiviudal, p: float, q: float, eps: float = 0.01) -> Indiviudal: 
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route.""" 
        # Replace visits within route
        n = max(0, min(np.random.geometric(1 - p) - 1, len(individual) - 1))
        indicies = sorted(np.random.choice(len(individual), n, replace = False))
        for i in indicies:
            # Replace visits through route.
            route_without_visit_i = individual.route[:i] + individual.route[i + 1:]
            # Actually replace the visit in the individual
            individual.route[i] = self.pick_visit_to_insert_based_on_SDR(route_without_visit_i, i)
            individual.mutation_step_sizes[i] = 0.5

        # Insert visists within the route the route.
        n = max(0, min(np.random.geometric(1 - q) - 1, len(individual)))
        indicies = sorted(np.random.choice(len(individual) + 1, n, replace = False))
        for i in reversed(indicies):
            best_visit = self.pick_visit_to_insert_based_on_SDR(individual.route, i)

            # Actually insert the information into the individual
            individual.route.insert(i, best_visit)
            individual.mutation_step_sizes.insert(i, 0.5)

        # Finally we perform multiple angle mutations
        return self.angle_mutation(individual, eps)

    def fix_length_optimized(self, individual: Indiviudal, indicies_to_skip: List[int] = None) -> Indiviudal:
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
                if len(individual) == 1:
                    return Indiviudal([], [])
                # Iteratively remove visits until the constraint is no longer violated, and update the distance saved throughout.
                if idx_skipped is None:
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

                else:
                    # Pop the corresponding elements
                    lengths_of_new_segments.pop(idx_skipped)
                    distances_saved.pop(idx_skipped)
                    sdsr_scores.pop(idx_skipped)

                    # Find the "updated" (i.e. after poping) indicies which needs updating 
                    m = len(individual)
                    if idx_skipped == 0:
                        indicies_which_needs_updating = [0] 
                    elif idx_skipped == m:
                        indicies_which_needs_updating = [m - 1] 
                    else:
                        indicies_which_needs_updating = [idx_skipped - 1, idx_skipped]

                    for idx in indicies_which_needs_updating:
                        # Recompute the lenghts of the new segments.
                        if idx == 0:
                            lengths_of_new_segments[0] = compute_length_of_relaxed_dubins_path(individual.states[1].angle_complement(), self.problem_instance.source, self.problem_instance.rho)
                            distances_saved[0] = (individual.segment_lengths[0] + individual.segment_lengths[1]) - lengths_of_new_segments[0]
                        elif idx == m - 1:
                            lengths_of_new_segments[m - 1] = compute_length_of_relaxed_dubins_path(individual.states[-2], self.problem_instance.sink, self.problem_instance.rho)
                            distances_saved[m - 1] = (individual.segment_lengths[-2] + individual.segment_lengths[-1]) - lengths_of_new_segments[-1]
                        else:
                            lengths_of_new_segments[idx] = dubins.shortest_path(individual.states[idx - 1].to_tuple(), individual.states[idx + 1].to_tuple(), self.problem_instance.rho).path_length()
                            distances_saved[idx] = (individual.segment_lengths[idx] + individual.segment_lengths[idx + 1]) - lengths_of_new_segments[idx]

                        # Update the associated information
                        sdsr_scores[idx] = individual.scores[idx] / distances_saved[idx]

                # This is according to the ratio: score of visit / length saved by excluding visit that is the "score distance saved ratio (SDSR)"
                idx_skipped = np.argmin(sdsr_scores)

                individual.route.pop(idx_skipped)
                individual.states.pop(idx_skipped)
                individual.scores.pop(idx_skipped)
                individual.mutation_step_sizes.pop(idx_skipped)
            
                # NOTE: we need to change both the segment length at the idx_skipped and idx_skipped + 1, since we are skipping the 
                #       visit at idx_skipped and this visit is partially responsible for these path segements
                individual.segment_lengths[idx_skipped + 1] = lengths_of_new_segments[idx_skipped]
                individual.segment_lengths.pop(idx_skipped)

                distance -= distances_saved[idx_skipped]

        return individual

    # TODO: At somepoint this should be converted to use corrolated mutations.
    def angle_mutation (self, individual: Indiviudal, eps: float) -> CEDOPADSRoute:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""

        # Core idea here is that in the start we are quite interested in making
        # large jumps in each psi and tau, however in the end we are more interested 
        # in wiggeling them around to perform the last bit of optimization
        mutation_step_size_mutation_var = 1 / np.sqrt(2 * len(individual.route))
        for idx in range(len(individual.route)):
            (k, _, tau, _) = individual.route[idx]
            individual.mutation_step_sizes[idx] *= np.exp(np.random.normal(0, mutation_step_size_mutation_var))
            new_tau = np.random.normal(tau, max(individual.mutation_step_sizes[idx], eps)) % (2 * np.pi)
            individual.route[idx] = (k, 0, new_tau, 0)

        return individual

    # ---------------------------------------- Parent Selection ----------------------------------------- #
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
        individual_with_highest_recorded_score = max(self.population, key = lambda individual: self.problem_instance.compute_score_of_route(individual.route, self.utility_function)) 
        highest_score_recorded = self.problem_instance.compute_score_of_route(individual_with_highest_recorded_score.route, self.utility_function)

        with tqdm(total=time_budget, desc="GA", disable = not display_progress_bar) as pbar:
            # Set initial progress bar information.
            while time.time() - start_time < time_budget:
                # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
                # and perform k-point crossover using these parents, to create a total of m^k children which is
                # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
                # the angles in order to find new versions of the parents with higher fitness values.
                offspring = []
                parents_to_be_mutated = [self.population[i] for i in self.stochastic_universal_sampling(cdf, self.xi)]
                for parent in parents_to_be_mutated:
                    mutated_parent = self.angle_mutation(parent, 0.01) 
                    fixed_mutated_parent = self.fix_length_optimized(mutated_parent) 
                    offspring.append(fixed_mutated_parent)

                while len(offspring) < self.lmbda - self.xi: 
                    parent_indicies = self.stochastic_universal_sampling(cdf, m = 2)
                    parents = [self.population[i] for i in parent_indicies]
                    for child in self.partially_mapped_crossover(parents): 
                        mutated_child = self.mutate(child, p = 0.2, q = 0.3) # NOTE: we need this deepcopy!
                        fixed_mutated_child = self.fix_length_optimized(mutated_child) # NOTE: This also could return the length of the mutated_child, which could be usefull in later stages.
                        offspring.append(fixed_mutated_child)
                
                # Survivor selection, responsible for find the lements which are passed onto the next generation.
                gen += 1
                self.population = self.mu_comma_lambda_selection(offspring)
                probabilities = self.sigma_scaling() 
                cdf = np.add.accumulate(probabilities)

                indiviudal_with_highest_score = max(self.population, key = lambda individual: self.problem_instance.compute_score_of_route(individual.route, self.utility_function)) 
                if self.problem_instance.compute_score_of_route(indiviudal_with_highest_score.route, self.utility_function) > highest_score_recorded:
                    individual_with_highest_recorded_score = indiviudal_with_highest_score


        return individual_with_highest_recorded_score.route

def _worker_function(data: Tuple[str, float, int, int, float]) -> float:
    file, time_budget, idx, seed, rho = data

    utility_function = utility_baseline
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file(file, needs_plotting = True)
    problem_instance.sensing_radii = (0, 0)
    problem_instance.rho = rho
    mu = 256
    ga = GA(problem_instance, utility_function, mu = mu, lmbda = mu * 7, xi = 64)
    ga.fitness_idx = idx

    np.random.seed(seed)
    random.seed(seed)

    route = ga.run(time_budget, display_progress_bar=False)
    return problem_instance.compute_score_of_route(route, utility_function)

def benchmark(time_budget: float = 120, n: int = 12):
    p4_files = ["p4.4.d.a.a.txt", "p4.4.h.a.a.txt", "p4.4.l.a.a.txt", "p4.4.p.a.a.txt"]
    p7_files = ["p7.4.d.a.a.txt", "p7.4.h.a.a.txt", "p7.4.l.a.a.txt", "p7.4.p.a.a.txt"]
    
#    number_of_cores = multiprocessing.cpu_count()
#    results = {idx: {"p4": {}, "p7": {}} for idx in range(4)}
#    for idx in range(4): 
#        with multiprocessing.Pool(number_of_cores) as pool: # TODO: change to number_of_cores
#            for file in p4_files:
#                print(idx, file)
#                arguments = [(file, time_budget, idx, seed, 2) for seed in range(n)]
#                results[idx]["p4"][file[5]] = list(pool.map(_worker_function, arguments))
#
#            for file in p7_files:
#                print(idx, file)
#                arguments = [(file, time_budget, idx, seed, 4) for seed in range(n)]
#                results[idx]["p7"][file[5]] = list(pool.map(_worker_function, arguments))

    # Save results
    import json
    #with open("results_DOP.json", "w+") as file:
    #    json.dump(results, file)
        
    with open("results_DOP.json", "r+") as file:
       results = json.load(file)

    plt.style.use("ggplot")
    width = 0.55
    x = np.arange(len(p4_files) + len(p7_files))

    tab10 = plt.get_cmap("tab10")
    problems = list(product(["p4", "p7"], ["d", "h", "l", "p"]))

    for idx in range(4):
        ys = [np.mean(results[str(idx)][p][letter]) for p, letter in problems]
        stds = [np.std(results[str(idx)][p][letter]) for p, letter in problems]

        x_idx = x - (idx - 3 / 2) * width / 4 if idx >= 2 else x + (idx + 1 / 2) * width / 4
        plt.errorbar(x_idx, ys, stds, linestyle="None", marker="o", color = tab10(idx))

    plt.xticks(ticks=x, labels=[f"{p}.{letter}" for p, letter in problems])
    plt.title('Comparison Between Fitness Functions')
    plt.xlabel("Problem")
    plt.ylabel("Mean Best Score & Standard Deviations")
    import matplotlib.lines as mlines
    indicators = [mlines.Line2D([], [], color=tab10(idx), label=f"$f_{idx}$", marker = "s") for idx in range(4)]
    plt.legend(handles=indicators, loc=1)

    # Show the plot
    plt.show()
        
if __name__ == "__main__":
    benchmark()