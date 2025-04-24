# %%
import dubins
from copy import deepcopy
import random
import time
from typing import List, Optional, Callable, Set, Tuple
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle,AreaOfAcquisition, Matrix, Position, Vector, compute_difference_between_angles
from numpy.typing import ArrayLike
import hashlib
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from library.CEDOPADS_solvers.local_search_operators import remove_visit, greedily_add_visits_while_posible, add_free_visits, get_equdistant_samples
from dataclasses import dataclass, field
from numba import njit

@dataclass()
class Individual:
    """Stores the information relating to an individual"""
    route: CEDOPADSRoute

    aoa_indicies: List[int]
    psi_mutation_step_sizes: List[float] 
    tau_mutation_step_sizes: List[float] 
    r_mutation_step_sizes: List[float] 

    # Optional fields 

    scores: List[float] = None
    states: List[State] = None 
    segment_lengths: List[float] = None 

    def __len__(self) -> int:
        """Returns the length of the route"""
        return len(self.route)

    def __repr__ (self) -> str:
        visits = [f"({int(k)}, {float(psi):.1f}, {float(tau):.1f}, {r:.1f})" for (k, psi, tau, r) in self.route]
        return "[" + " ".join(visits) + "]"

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

class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem_instance: CEDOPADSInstance
    utility_function: UtilityFunction
    mu: int # Population Number
    lmbda: int # Number of offspring
    number_of_nodes: int
    node_indicies: Set[int]
    node_indicies_array: Vector

    def __init__ (self, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, mu: int = 256, lmbda: int = 256 * 7, xi: int = 0, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        self.problem_instance = problem_instance
        self.utility_function = utility_function
        self.lmbda = lmbda
        self.mu = mu
        self.xi = xi
        self.number_of_nodes = len(self.problem_instance.nodes)
        self.node_indicies = set(range(self.number_of_nodes))
        self.node_indicies_arrray = np.array(range(self.number_of_nodes))
        
        self.r_default_mutation_step_size = (self.problem_instance.sensing_radii[1] - self.problem_instance.sensing_radii[0]) / 256
        self.tau_default_mutation_step_size = self.problem_instance.eta / 256 

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

            aoa_indicies = []
            while self.problem_instance.is_route_feasable(route):
                # Pick a new random visit and add it to the route.
                k = random.choice(list(self.node_indicies.difference(blacklist)))
                blacklist.add(k)
                aoa_indicies.append(np.random.randint(len(self.problem_instance.nodes[k].AOAs)))
                psi = self.problem_instance.nodes[k].AOAs[aoa_indicies[-1]].generate_uniform_angle() 
                tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
                r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

                route.append((k, psi, tau, r)) 

            # Initialize what we need to know from the individual
            scores = self.problem_instance.compute_scores_along_route(route, self.utility_function)

            psi_mutation_step_sizes = [self.problem_instance.nodes[k].get_AOA(psi).phi / 256 for (k, psi, _, _) in route]
            tau_mutation_step_sizes = [self.tau_default_mutation_step_size for (_, _, _, _) in route]
            r_mutation_step_sizes = [self.r_default_mutation_step_size for (_, _, _, _) in route]

            population.append(Individual(route, scores = scores,
                                         aoa_indicies = aoa_indicies,
                                         psi_mutation_step_sizes = psi_mutation_step_sizes, 
                                         tau_mutation_step_sizes = tau_mutation_step_sizes,
                                         r_mutation_step_sizes = r_mutation_step_sizes,
                                         states = self.problem_instance.get_states(route))) 

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

        (i, j) = tuple(sorted(np.random.choice(m, 2, replace = False)))

        offspring = []
        for paternal, maternal in [(parents[0], parents[1]), (parents[1], parents[0])]:

            route = paternal.route[:i] + maternal.route[i:j] + paternal.route[j:]
            psi_mutation_step_sizes = paternal.psi_mutation_step_sizes[:i] + maternal.psi_mutation_step_sizes[i:j] + paternal.psi_mutation_step_sizes[j:]
            tau_mutation_step_sizes = paternal.tau_mutation_step_sizes[:i] + maternal.tau_mutation_step_sizes[i:j] + paternal.tau_mutation_step_sizes[j:]
            r_mutation_step_sizes = paternal.r_mutation_step_sizes[:i] + maternal.r_mutation_step_sizes[i:j] + paternal.r_mutation_step_sizes[j:]
            aoa_indicies = paternal.aoa_indicies[:i] + maternal.aoa_indicies[i:j] + paternal.aoa_indicies[j:]
            states = paternal.states[:i] + maternal.states[i:j] + paternal.states[j:]
            scores = paternal.scores[:i] + maternal.scores[i:j] + paternal.scores[j:]


            # Fix any mishaps when the child was created, i.e. make sure that the same node i never visited twice.
            ks = {route[idx][0] for idx in range(i, j + 1)} 

            # NOTE: Normaly the order of the visits is reversed, however since our distance "metric" is in this case non-symetric we will not use this convention.
            indicies_which_may_replace = [idx for idx in range(i, j) if paternal.route[idx][0] not in ks]
            number_of_visits_replaced, number_of_visits_which_can_be_replaced = 0, len(indicies_which_may_replace)

            indicies_to_remove = []
            for idx in itertools.chain(range(i), range(j, len(route))):
                # Replace the visit with one from the other parent if posible otherwise simply remove the visit entirely
                if route[idx][0] in ks:
                    if number_of_visits_replaced < number_of_visits_which_can_be_replaced:
                        route[idx] = paternal.route[indicies_which_may_replace[number_of_visits_replaced]]
                        psi_mutation_step_sizes[idx] = paternal.psi_mutation_step_sizes[indicies_which_may_replace[number_of_visits_replaced]]
                        tau_mutation_step_sizes[idx] = paternal.tau_mutation_step_sizes[indicies_which_may_replace[number_of_visits_replaced]]
                        r_mutation_step_sizes[idx] = paternal.r_mutation_step_sizes[indicies_which_may_replace[number_of_visits_replaced]]
                        aoa_indicies[idx] = paternal.aoa_indicies[indicies_which_may_replace[number_of_visits_replaced]]
                        states[idx] = paternal.states[indicies_which_may_replace[number_of_visits_replaced]]
                        scores[idx] = paternal.scores[indicies_which_may_replace[number_of_visits_replaced]]
                        number_of_visits_replaced += 1
                    else:
                        indicies_to_remove.append(idx)

            # actually remove the indices, we have to do it to not mess with the indicies.
            for idx in reversed(indicies_to_remove):
                route.pop(idx)
                psi_mutation_step_sizes.pop(idx)
                tau_mutation_step_sizes.pop(idx)
                r_mutation_step_sizes.pop(idx)
                states.pop(idx)
                scores.pop(idx)
                aoa_indicies.pop(idx)

            offspring.append(Individual(route, aoa_indicies, psi_mutation_step_sizes, tau_mutation_step_sizes, r_mutation_step_sizes, scores, states)) 

        return offspring

    #--------------------------------------------- Mutation Operators -------------------------------------- 
    def mutate(self, individual: Individual, p: float, q: float, eps: float = 0.001) -> Individual: 
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route.""" 
        # Replace visits within route
        n = max(0, min(np.random.geometric(1 - p) - 1, len(individual) - 1))
        indicies = sorted(np.random.choice(len(individual), n, replace = False))
        for i in indicies:
            # Replace visits through route.
            route_without_visit_i = individual.route[:i] + individual.route[i + 1:]
            best_visit, score = self.pick_visit_to_insert_based_on_SDR(route_without_visit_i, i)
            

            # Actually replace the visit in the individual
            individual.route[i] = best_visit
            aoa = self.problem_instance.nodes[best_visit[0]].get_AOA(best_visit[1])

            individual.psi_mutation_step_sizes[i] = aoa.phi / 256
            individual.tau_mutation_step_sizes[i] = self.tau_default_mutation_step_size 
            individual.r_mutation_step_sizes[i] = self.r_default_mutation_step_size

            individual.aoa_indicies[i] = self.problem_instance.nodes[best_visit[0]].AOAs.index(aoa)

            individual.states[i] = self.problem_instance.get_state(best_visit)
            individual.scores[i] = score

        indicies_to_skip = indicies

        # Insert visists within the route the route.
        n = max(0, min(np.random.geometric(1 - q) - 1, len(individual)))
        indicies = sorted(np.random.choice(len(individual) + 1, n, replace = False))
        for j, i in enumerate(reversed(indicies), 1):
            best_visit, score = self.pick_visit_to_insert_based_on_SDR(individual.route, i)

            # Actually insert the information into the individual
            individual.route.insert(i, best_visit)
            individual.scores.insert(i, score)
            individual.states.insert(i, self.problem_instance.get_state(best_visit))

            aoa = self.problem_instance.nodes[best_visit[0]].get_AOA(best_visit[1])

            individual.psi_mutation_step_sizes.insert(i,  aoa.phi / 256)
            individual.tau_mutation_step_sizes.insert(i, self.tau_default_mutation_step_size)
            individual.r_mutation_step_sizes.insert(i, self.r_default_mutation_step_size)
            
            individual.aoa_indicies.insert(i, self.problem_instance.nodes[best_visit[0]].AOAs.index(aoa))
            indicies_to_skip.append(i + n - j)

        # Finally we perform multiple angle mutations
        #return self.non_self_adaptive_scalar_mutation(individual), indicies_to_skip
        return self.self_adaptive_scalar_mutation(individual), indicies_to_skip

    def pick_best_explored_visit_based_on_sdr(self, route: CEDOPADSRoute, i: int, k: int) -> Tuple[Visit, float, float]:
        """Picks the best visit to node k based on the SDR, by inserting it within path segment i."""
        best_visit = None
        score_of_best_visit = None
        sdr_of_best_visit = 0 
        # If we have not yet seen the node before, pre compute everything that will be needed.
        if i == 0:
            # Compute the original distance before including the visit
            q = self.problem_instance.get_state(route[0])
            original_distance = compute_length_of_relaxed_dubins_path(q.angle_complement(), self.problem_instance.source, self.problem_instance.rho)
            for visit, q_visit, score in zip(self.explored_visits[k], self.explored_states[k], self.explored_scores[k]):
                # Compute score associated with visit and the distance through the visit
                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_visit.angle_complement(), self.problem_instance.source, self.problem_instance.rho) + 
                                          dubins.shortest_path(q_visit.to_tuple(), q.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    return visit, score

                elif (sdr_of_visit := score / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit
                    score_of_best_visit = score

        elif i == len(route):
            # Compute the original distance before including the visit
            q = self.problem_instance.get_state(route[-1])
            original_distance = compute_length_of_relaxed_dubins_path(q, self.problem_instance.sink, self.problem_instance.rho)
            for visit, q_visit, score in zip(self.explored_visits[k], self.explored_states[k], self.explored_scores[k]):
                # Compute score associated with visit and the distance through the visit
                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_visit, self.problem_instance.sink, self.problem_instance.rho) +
                                          dubins.shortest_path(q.to_tuple(), q_visit.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    return visit, score

                elif (sdr_of_visit := score / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit
                    score_of_best_visit = score

        else: 
            # Compute the original distance before including the visit
            q_i, q_t = self.problem_instance.get_state(route[i - 1]), self.problem_instance.get_state(route[i])
            original_distance = dubins.shortest_path(q_i.to_tuple(), q_t.to_tuple(), self.problem_instance.rho).path_length()
            for visit, q_visit, score in zip(self.explored_visits[k], self.explored_states[k], self.explored_scores[k]):
                # Compute score associated with visit and the distance through the visit
                distance_through_visit = (dubins.shortest_path(q_i.to_tuple(), q_visit.to_tuple(), self.problem_instance.rho).path_length() + 
                                          dubins.shortest_path(q_visit.to_tuple(), q_t.to_tuple(), self.problem_instance.rho).path_length())

                if distance_through_visit == original_distance:
                    return visit, score

                elif (sdr_of_visit := score / (distance_through_visit - original_distance)) > sdr_of_best_visit:
                    best_visit = visit
                    sdr_of_best_visit = sdr_of_visit
                    score_of_best_visit = score
    
        return best_visit, sdr_of_best_visit, score_of_best_visit

    def pick_visit_to_insert_based_on_SDR (self, route: CEDOPADSRoute, i: int, m: int = 3) -> Tuple[Visit, float]:
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
        ks = np.random.choice(self.node_indicies_arrray[self.mask_for_overwrite], size = m, p = probs)

        # Find the actual best visit for visiting node k according to the SDR score.
        best_visits_and_information = [self.pick_best_explored_visit_based_on_sdr(route, i, k) for k in ks]

        idx = np.argmax([sdr for (_, sdr, _) in best_visits_and_information])
        return best_visits_and_information[idx][0], best_visits_and_information[idx][2]

    def self_adaptive_scalar_mutation (self, individual: Individual, eps: float = 0.01, c_psi: float = 1.0, c_tau: float = 1.0, c_r: float = 1.0) -> Individual:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
        # 1. Update mutation step sizes.
        baseline_scale_for_step_sizes = 1 / np.sqrt(2 * len(individual))

        n = max(0, min(np.random.geometric(1 - 0.2) - 1, len(individual)))
        indicies = np.random.choice(len(individual), n, replace = False)
        
        mutation_steps_for_mutation_step_sizes = np.random.normal(0, 1, size=(n, 3))
        mutation_steps_for_mutation_step_sizes[:, 0] *= c_psi * baseline_scale_for_step_sizes
        mutation_steps_for_mutation_step_sizes[:, 1] *= c_tau * baseline_scale_for_step_sizes
        mutation_steps_for_mutation_step_sizes[:, 2] *= c_r * baseline_scale_for_step_sizes

        for i, j in enumerate(indicies):
            individual.psi_mutation_step_sizes[j] = np.maximum(individual.psi_mutation_step_sizes[j] * np.exp(mutation_steps_for_mutation_step_sizes[i, 0]), eps)
            individual.tau_mutation_step_sizes[j] = np.maximum(individual.tau_mutation_step_sizes[j] * np.exp(mutation_steps_for_mutation_step_sizes[i, 1]), eps)
            individual.r_mutation_step_sizes[j] = np.maximum(individual.r_mutation_step_sizes[j] * np.exp(mutation_steps_for_mutation_step_sizes[i, 2]), eps)

        # TODO: Mutation steps
        N_psi = np.random.normal(0, [individual.psi_mutation_step_sizes[j] for j in indicies])
        N_tau = np.random.normal(0, [individual.tau_mutation_step_sizes[j] for j in indicies])
        N_r = np.random.normal(0, [individual.r_mutation_step_sizes[j] for j in indicies])

        # Update each visit accordingly.
        for i, j in enumerate(indicies):
            
            # Update psi, by skipping to the next aoa clockwise / counter clockwise if we get out of bounds.
            (k, psi, tau, r) = individual.route[j]
            new_psi = psi + N_psi[i]
            aoa = self.problem_instance.nodes[k].AOAs[individual.aoa_indicies[j]]
            if not aoa.contains_angle(new_psi):
                # Move counterclockwise if N_psi > 0 and clockwise otherwise.
                change_in_aoa_idx = 1 if N_psi[i] > 0 else -1 
                individual.aoa_indicies[j] = (individual.aoa_indicies[j] + change_in_aoa_idx) % len(self.problem_instance.nodes[k].AOAs) 
                new_aoa = self.problem_instance.nodes[k].AOAs[individual.aoa_indicies[j]]

                # Pick the angle closest to the original angle, from the new aoa
                new_psi = new_aoa.a if change_in_aoa_idx == 1 else new_aoa.b
                delta_psi = new_psi - psi

            else:
                delta_psi = N_psi[i]

            # Update tau but make sure that target k remains within the observation cone
            new_tau = (tau + N_tau[i] + delta_psi) % (2 * np.pi)
            if compute_difference_between_angles(new_tau, (new_psi + np.pi) % (2 * np.pi)) > self.problem_instance.eta / 2:
                # If this is not the case, simply move tau the same amount as N_tau.
                new_tau = (tau + delta_psi) % (2 * np.pi) 

            # Update r but make sure that it remains within the interval [r_min; r_max]
            new_r = max(min(r + N_r[i], self.problem_instance.sensing_radii[0]), self.problem_instance.sensing_radii[1])

            individual.route[j] = (k, new_psi, new_tau, new_r)
            individual.states[j] = self.problem_instance.get_state(individual.route[j])
            individual.scores[j] = self.problem_instance.compute_score_of_visit(individual.route[j], self.utility_function)

            individual.segment_lengths = None # Signals that they need to be recomputed.

        return individual

    # --------------------------------------------------------- Feasability ------------------------------------------------------------------- #
    def fix_length_optimized(self, individual: Individual, indicies_to_skip: List[int]) -> Individual:
        """Removes visits from the individual until it obeys the distance constraint."""
        indicies_to_skip_mask = [np.inf if idx in indicies_to_skip else 0 for idx in range(len(individual))]

        # Keep track of these rather than recomputing them on the fly
        if individual.segment_lengths is None:
            individual.segment_lengths = self.problem_instance.compute_lengths_of_route_segments_from_states(individual.states)

        distance = sum(individual.segment_lengths) 

        if distance > self.problem_instance.t_max:
            # Compute sdsr "score distance saved ratio" of each visit these will be updated throughout
            idx_skipped = None
            while distance > self.problem_instance.t_max: 
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
                idx_skipped = np.argmin([sdsr_score + mask_score for sdsr_score, mask_score in zip(sdsr_scores, indicies_to_skip_mask)])

                indicies_to_skip_mask.pop(idx_skipped)
                individual.route.pop(idx_skipped)
                individual.states.pop(idx_skipped)
                individual.scores.pop(idx_skipped)
                individual.psi_mutation_step_sizes.pop(idx_skipped)
                individual.tau_mutation_step_sizes.pop(idx_skipped)
                individual.r_mutation_step_sizes.pop(idx_skipped)
                individual.aoa_indicies.pop(idx_skipped)
            
                # NOTE: we need to change both the segment length at the idx_skipped and idx_skipped + 1, since we are skipping the 
                #       visit at idx_skipped and this visit is partially responsible for these path segements
                individual.segment_lengths[idx_skipped + 1] = lengths_of_new_segments[idx_skipped]
                individual.segment_lengths.pop(idx_skipped)

                distance -= distances_saved[idx_skipped]

        return individual

    # ----------------------------------------- Local Search Operators ------------------------------------- #
    def optimize_visits(self, individual: Individual, n_iterations: int = 8, maximum_number_of_attemps: int = 4) -> Individual:
        """Optimizes the visits of the individual, in an iterative manor."""
        # If the segment lengths of the individual has not yet been computed, compute them.
        if individual.segment_lengths is None:
            individual.segment_lengths = self.problem_instance.compute_lengths_of_route_segments_from_states(individual.states)

        # At each iteration we are allowed extend the route by a distance of t_remaining / n_iterations, if it increases the SDR or simply the score.
        t_remaining = self.problem_instance.t_max - sum(individual.segment_lengths)
        if t_remaining < 0: # Do if we dont have the budget nothing
            return individual

        inverse_sdr_scores = np.array([(in_bound_dist + out_bound_dist) / (score + 0.001) for score, in_bound_dist, out_bound_dist in zip(individual.scores, individual.segment_lengths[:-1], individual.segment_lengths[1:])])
        inverse_ratio_of_maximal_scores = np.array([self.problem_instance.nodes[k].base_line_score / (score + 0.001) for score, (k, _, _, _) in zip(individual.scores, individual.route)])

        for i in range(n_iterations):
            # We want to pick the visits with either the lowest score ratio (i.e. the visits where the score can be increased the most or the visits where the SDR score is lowest)
            probs = (inverse_ratio_of_maximal_scores / np.sum(inverse_ratio_of_maximal_scores) + (inverse_sdr_scores / np.sum(inverse_sdr_scores))) / 2
            if any([p < 0 for p in probs]):
                print(i, probs, individual.scores)

            idx = np.random.choice(len(individual), p = probs)
            (k, psi, tau, r) = individual.route[idx]

            for _ in range(maximum_number_of_attemps):
                # Generate new psi, tau & r
                N_psi = np.random.normal(0, scale = individual.psi_mutation_step_sizes[idx])
                N_tau = np.random.normal(0, scale = individual.tau_mutation_step_sizes[idx])
                new_psi = psi + N_psi
                aoa = self.problem_instance.nodes[k].AOAs[individual.aoa_indicies[idx]]
                if not aoa.contains_angle(new_psi):
                    # Move counterclockwise if N_psi > 0 and clockwise otherwise.
                    change_in_aoa_idx = 1 if N_psi > 0 else -1 
                    new_aoa_index = (individual.aoa_indicies[idx] + change_in_aoa_idx) % len(self.problem_instance.nodes[k].AOAs) 
                    new_aoa = self.problem_instance.nodes[k].AOAs[new_aoa_index]

                    # Pick the angle closest to the original angle, from the new aoa
                    new_psi = new_aoa.a if change_in_aoa_idx == 1 else new_aoa.b
                    delta_psi = new_psi - psi

                else:
                    new_aoa_index = None
                    delta_psi = N_psi

                new_tau = (tau + N_tau + delta_psi) % (2 * np.pi)
                if compute_difference_between_angles(new_tau, (new_psi + np.pi) % (2 * np.pi)) > self.problem_instance.eta / 2:
                    # If this is not the case, simply move tau the same amount as N_tau.
                    new_tau = (tau + delta_psi) % (2 * np.pi) 

                # Update r but make sure that it remains within the interval [r_min; r_max]
                new_r = max(min(r + np.random.normal(0, individual.r_mutation_step_sizes[idx]), self.problem_instance.sensing_radii[0]), self.problem_instance.sensing_radii[1])

                # Generate new visit and state 
                new_visit = (k, new_psi, new_tau, new_r)
                new_q = self.problem_instance.get_state(new_visit)
                
                # Compute new lengths.
                if len(individual) == 1:
                    new_lengths = (compute_length_of_relaxed_dubins_path(new_q.angle_complement(), self.problem_instance.source, self.problem_instance.rho),
                                   compute_length_of_relaxed_dubins_path(new_q, self.problem_instance.sink, self.problem_instance.rho))

                elif idx == 0:
                    new_lengths = (compute_length_of_relaxed_dubins_path(new_q.angle_complement(), self.problem_instance.source, self.problem_instance.rho),
                                   dubins.shortest_path(new_q.to_tuple(), individual.states[idx + 1].to_tuple(), self.problem_instance.rho).path_length())

                elif idx == len(individual) - 1:
                    new_lengths = (dubins.shortest_path(individual.states[idx - 1].to_tuple(), new_q.to_tuple(), self.problem_instance.rho).path_length(),
                                   compute_length_of_relaxed_dubins_path(new_q, self.problem_instance.sink, self.problem_instance.rho))

                else:
                    new_lengths = (dubins.shortest_path(individual.states[idx - 1].to_tuple(), new_q.to_tuple(), self.problem_instance.rho).path_length(),
                                   dubins.shortest_path(new_q.to_tuple(), individual.states[idx + 1].to_tuple(), self.problem_instance.rho).path_length())

                # Compute new scores & the total change in length.
                new_score = self.problem_instance.compute_score_of_visit(new_visit, self.utility_function) 
                change_in_length = (new_lengths[0] + new_lengths[1]) - (individual.segment_lengths[idx] + individual.segment_lengths[idx + 1])

                if (change_in_length < t_remaining / n_iterations) and (new_score / sum(new_lengths) > (1 / inverse_sdr_scores[idx])) or (new_score > individual.scores[idx]):
                    # Udate individual
                    individual.states[idx] = new_q
                    individual.route[idx] = new_visit

                    individual.scores[idx] = new_score
                    individual.segment_lengths[idx] = new_lengths[0]
                    individual.segment_lengths[idx + 1] = new_lengths[1]
                    if new_aoa_index is not None:
                        individual.aoa_indicies[idx] = new_aoa_index
                    
                    # Update SDR scores & ratios of maximal scores.
                    inverse_ratio_of_maximal_scores[idx] = self.problem_instance.nodes[k].base_line_score / new_score
                    inverse_sdr_scores[idx] = (new_lengths[0] + new_lengths[1]) / new_score
                    
                    if idx == 0 and len(individual) >= 2: 
                        # Update indicies after
                        inverse_sdr_scores[idx + 1] = (new_lengths[1] + individual.segment_lengths[idx + 2]) / individual.scores[idx + 1]
                        
                    elif idx == len(individual) - 1 and len(individual) >= 2: 
                        # Update indicies before
                        inverse_sdr_scores[idx - 1] = (individual.segment_lengths[idx - 1] + new_lengths[0]) / individual.scores[idx - 1]

                    elif len(individual) >= 3:
                        # Update indicies before and after
                        inverse_sdr_scores[idx - 1] = (individual.segment_lengths[idx - 1] + new_lengths[0])  / individual.scores[idx - 1]
                        inverse_sdr_scores[idx + 1] = (new_lengths[1] + individual.segment_lengths[idx + 2])  / individual.scores[idx + 1]

                    break

        return individual

    def add_visits(self, individual: Individual, maximum_number_of_attemps: int = 0) -> Individual:
        """Tries to add close to free visits to the individual, and checks if the SDR score increases."""
        if individual.segment_lengths is None:
            individual.segment_lengths = self.problem_instance.compute_lengths_of_route_segments(individual.route)

        total_score = sum(individual.scores)
        ks_already_visited = {k for (k, _, _, _) in individual.route}
        for i in range(maximum_number_of_attemps):
            # 1. Pick a dubins path segement (it is probabily suitable to pick one of the longer segments.)
            total_length = sum(individual.segment_lengths)
            if i == 0:
                sdr_score = total_score / total_length

            probs_for_segments = [segment_length / total_length for segment_length in individual.segment_lengths]
            idx = np.random.choice(len(individual) + 1, p = probs_for_segments)

            # 2. Compute the minimum distance from each point to the dubins path segment
            ks = [k for k in range(len(self.problem_instance.nodes)) if not (k in ks_already_visited)] 

            minimum_distances = np.zeros(len(ks))

            for j, k in enumerate(ks):
                minimum_distances[j] = 0
            
            # TODO: Compute these based on the minimum distance to the existing path segment & the score of the node.
            probs_for_ks = (1 / len(ks)) * np.ones(len(ks))

            # 3. Based on the minimum distance from the point to the dubins path segment, try to add a visit to the route.
            #    if the sdr does not increase, then don't add the visit to the route.
            k = ks[np.random.choice(len(ks), p = probs_for_ks)]
            new_visit = self.pick_best_explored_visit_based_on_sdr(individual.route, i, k)[0]
            new_state = self.problem_instance.get_state(new_visit)

            # Compute the lengths of the new path segments.
            if idx == 0:
                new_lengths = (compute_length_of_relaxed_dubins_path(new_state.angle_complement(), self.problem_instance.source, self.problem_instance.rho),
                               dubins.shortest_path(new_state.to_tuple(), individual.states[idx].to_tuple(), self.problem_instance.rho).path_length())

            elif idx == len(individual):
                new_lengths = (dubins.shortest_path(individual.states[-1].to_tuple(), new_state.to_tuple(), self.problem_instance.rho).path_length(),
                               compute_length_of_relaxed_dubins_path(new_state, self.problem_instance.sink, self.problem_instance.rho))

            else:
                new_lengths = (dubins.shortest_path(individual.states[idx - 1].to_tuple(), new_state.to_tuple(), self.problem_instance.rho).path_length(),
                               dubins.shortest_path(new_state.to_tuple(), individual.states[idx].to_tuple(), self.problem_instance.rho).path_length())

            
            # 4. Add visit to route if the sdr score is increased.
            score_of_new_visit = self.problem_instance.compute_score_of_visit(new_visit, self.utility_function)
            change_in_length = (new_lengths[0] + new_lengths[1]) - individual.segment_lengths[idx]
            
            if (new_sdr_score := (total_score + score_of_new_visit) / (total_length + change_in_length)) >= sdr_score:
                # Update information used within the method.
                sdr_score = new_sdr_score
                total_score += score_of_new_visit
                ks_already_visited.add(k)
                
                # Set information within the individual
                individual.segment_lengths[idx] = new_lengths[0]
                individual.segment_lengths.insert(idx + 1, new_lengths[1])
                individual.route.insert(idx, new_visit)
                individual.scores.insert(idx, score_of_new_visit) 
                individual.states.insert(idx, new_state)

                aoa = self.problem_instance.nodes[new_visit[0]].get_AOA(new_visit[1])
                individual.aoa_indicies.insert(idx, self.problem_instance.nodes[new_visit[0]].AOAs.index(aoa))

                individual.psi_mutation_step_sizes.insert(idx,  aoa.phi / 256)
                individual.tau_mutation_step_sizes.insert(idx, self.tau_default_mutation_step_size)
                individual.r_mutation_step_sizes.insert(idx, self.r_default_mutation_step_size)

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

    def mu_plus_lambda_selection(self, offspring: List[CEDOPADSRoute]) -> List[CEDOPADSRoute]:
        """Merges the newly generated offspring with the generation and selects the best mu individuals to survive until the next generation."""
        # NOTE: Recall that the fitness is time dependent hence we need to recalculate it for the parents.
        fitnesses_of_parents_and_offspring = np.array(list(map(lambda child: self.fitness_function(child), self.population + offspring)))

        # Calculate the indicies of the best performing memebers
        indicies_of_new_generation = np.argsort(fitnesses_of_parents_and_offspring)[-self.mu:]
        
        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_parents_and_offspring[indicies_of_new_generation]

        return [self.population[i] if i < self.mu else offspring[i - self.mu] for i in indicies_of_new_generation]

    # -------------------------------------------- Main Loop of GA --------------------------------------- #
    def run(self, time_budget: float, display_progress_bar: bool = False, trace: bool = False) -> CEDOPADSRoute:
        """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
        if trace:
            info_mean_fitness = []
            info_min_fitness = []
            info_max_fitness = []
            info_goat_fitness = []

        start_time = time.time()
        # Generate an initial population, which almost satifies the distant constraint, ie. add nodes until
        # the sink cannot be reached within d - t_max units where d denotes the length of the route.
        self.population = self.initialize_population()

        # These are used within the pick visit based on SDR functions to save time by simply performing lookups
        self.explored_visits = {}
        self.explored_states = {} 
        self.explored_scores = {}

        for k in range(len(self.problem_instance.nodes)):
            self.explored_visits[k] = list(get_equdistant_samples(self.problem_instance, k))
            self.explored_states[k] = self.problem_instance.get_states(self.explored_visits[k])
            self.explored_scores[k] = self.problem_instance.compute_scores_along_route(self.explored_visits[k], self.utility_function)

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
        probabilities = self.windowing()
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
            
            if trace:
                info_mean_fitness.append(np.mean(self.fitnesses))
                info_min_fitness.append(np.min(self.fitnesses))
                info_max_fitness.append(np.max(self.fitnesses))
                info_goat_fitness.append(self.fitnesses[idx_of_individual_with_highest_fitness])

            while time.time() - start_time < time_budget:
                # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
                # and perform k-point crossover using these parents, to create a total of m^k children which is
                # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
                # the angles in order to find new versions of the parents with higher fitness values.
                
                offspring = []
                while len(offspring) < self.lmbda - self.xi: 
                    parent_indicies = self.stochastic_universal_sampling(cdf, m = 2)
                    parents = [self.population[i] for i in parent_indicies]
                    for child in self.partially_mapped_crossover(parents):
                        mutated_child, indicies_to_skip = self.mutate(child, p = 0.1, q = 0.3) 
                        fixed_mutated_child = self.fix_length_optimized(self.optimize_visits(self.add_visits(mutated_child)), indicies_to_skip) 
                        offspring.append(fixed_mutated_child)
                
                # Survivor selection, responsible for find the lements which are passed onto the next generation.
                gen += 1
                self.population = self.mu_comma_lambda_selection(offspring)
                probabilities = self.sigma_scaling() 
                
                # Check for premature convergence.
                if np.isnan(probabilities).any():
                    print("The genetic algorithm converged prematurely.")
                    return self.individual_with_highest_recorded_fitness.route

                cdf = np.add.accumulate(probabilities)

                idx_of_individual_with_highest_fitness = np.argmax(self.fitnesses)
                if self.fitnesses[idx_of_individual_with_highest_fitness] > self.highest_fitness_recorded:
                    self.individual_with_highest_recorded_fitness = deepcopy(self.population[idx_of_individual_with_highest_fitness])
                    self.highest_fitness_recorded = self.fitnesses[idx_of_individual_with_highest_fitness]
                    self.length_of_individual_with_highest_recorded_fitness = self.problem_instance.compute_length_of_route(self.individual_with_highest_recorded_fitness.route)

                #assert sum(self.individual_with_highest_recorded_fitness.scores) == self.highest_fitness_recorded

                #for individual in self.population:
                #    assert len(individual.route) == len({k for (k, _, _, _) in individual.route})

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                if display_progress_bar:
                    pbar.n = round(time.time() - start_time, 1)
                    avg_fitness = np.mean(self.fitnesses)
                    avg_distance = np.mean([sum(individual.segment_lengths) for individual in self.population])
                    length_of_best_route = sum(self.individual_with_highest_recorded_fitness.segment_lengths) 
                    avg_length = sum(len(route) for route in self.population) / self.mu
                    pbar.set_postfix({
                            "Gen": gen,
                            "Best": f"({self.highest_fitness_recorded:.1f}, {self.length_of_individual_with_highest_recorded_fitness:.1f}, {len(self.individual_with_highest_recorded_fitness)})",
                            "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})",
                            "Var": f"{np.var(self.fitnesses):.1f}"
                    })

                if trace:
                    info_mean_fitness.append(np.mean(self.fitnesses))
                    info_min_fitness.append(np.min(self.fitnesses))
                    info_max_fitness.append(np.max(self.fitnesses))
                    info_goat_fitness.append(self.highest_fitness_recorded)
            
        # Plot information if a trace is requested.
        if trace:
            plt.style.use("ggplot")
            xs = list(range(gen + 1))
            plt.fill_between(xs, info_min_fitness, info_max_fitness, label="Fitnesses", color="tab:blue", alpha=0.6)
            plt.plot(xs, info_mean_fitness, color="tab:blue", label="Mean Fitness")
            plt.plot(xs, info_goat_fitness, color="black", label="GOAT fitness")
            plt.legend()
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.show()

        return self.individual_with_highest_recorded_fitness.route

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.g.c.a.txt", needs_plotting = True)
    mu = 256
    ga = GA(problem_instance, utility_function, mu = mu, lmbda = mu * 7, xi = int(np.floor(mu / 4)))
    #import cProfile
    #cProfile.run("ga.run(300, display_progress_bar = True)", sort = "cumtime")
    route = ga.run(300, display_progress_bar=True, trace=True)
    augmented_route = add_free_visits(problem_instance, route, utility_function)
    print(f"Score of augmented route: {problem_instance.compute_score_of_route(augmented_route, utility_function)} from {problem_instance.compute_score_of_route(route, utility_function)}")
    problem_instance.plot_with_route(route, utility_function)
    plt.show()
# %%
