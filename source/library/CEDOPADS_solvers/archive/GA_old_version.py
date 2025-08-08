# %%
import dubins
from copy import deepcopy
import random
import cProfile
import time
from typing import List, Tuple, Optional, Callable
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute, Visit
from library.core.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle, AngleInterval, Matrix, Position
from numpy.typing import ArrayLike
from classes.node import Node
import hashlib
from tqdm import tqdm
from itertools import combinations_with_replacement, combinations
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import truncnorm

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

                distance = area / np.linalg.norm(positions[i] - positions[j]) + 0.01 
                # NOTE: the 0.01 is used in case the area is 0, which happends if the points at positions i,j and k are colinear
                tensor[i, j, k] = scores[k] ** c_s / distance ** c_d # NOTE: The 0.01 is used to prevent nummerical errors, if we get a 0 we might divide by inf.
                tensor[j, i, k] = tensor[i, j, k]

    return tensor

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
        self.number_of_nodes = len(self.problem.nodes)
        self.node_indices = set(range(self.number_of_nodes))
        self.node_indices_array = np.array(range(self.number_of_nodes))
        
        # Compute distances which will be used to guide the mutation operator
        positions = np.concat((np.array([node.pos for node in problem.nodes]), problem.source.reshape(1, 2), problem.sink.reshape(1, 2)))
        self.sdr_tensor = compute_sdr_tensor(positions, np.array([node.score for node in problem.nodes]), c_s = 2.0, c_d = 2.0)
        self.sdrs_for_overwrite = np.empty_like(self.sdr_tensor[0][0]) 
        self.default_mask = np.ones(self.number_of_nodes, dtype=bool)
        self.mask_for_overwrite = np.empty_like(self.default_mask)

        # Seed RNG for repeatability
        if seed is None:
            hash_of_id = hashlib.sha1(problem.problem_id.encode("utf-8")).hexdigest()[:8]
            seed = int(hash_of_id, 16)
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Make sure that the sink can be reached from the source otherwise there is no point in running the algorithm.
        if (d := np.linalg.norm(self.problem.source - self.problem.sink)) > self.t_max:
            raise ValueError(f"Cannot construct a valid dubins route since ||source - sink|| = {d} is greater than t_max = {self.t_max}")

    def pick_new_visit(self, individual: CEDOPADSRoute) -> Visit:
        """Picks a new place to visit, according to how much time is left to run the algorithm."""
        # For the time being simply pick a random node to visit FIXME: Should be updated, 
        # maybe we pick them with a probability which is propotional to the distance to the individual?
        blacklist = set(k for (k, _, _) in individual)

        k = random.choice(list(self.node_indices.difference(blacklist)))
        psi = random.choice(self.problem.nodes[k].intervals).generate_uniform_angle()
        tau = (psi + np.pi + np.random.uniform( - self.eta / 2, self.eta / 2)) % (2 * np.pi)

        return (k, psi, tau)

    def pick_new_visit_based_on_sdr(self, individual: CEDOPADSRoute, i: int, j: int) -> Visit:
        """Generates a visit, based on the distance from the straight line segment between node i and node j."""
        self.sdrs_for_overwrite[:] = self.sdr_tensor[i][j][:]
        #distances = self.distance_tensor[i][j] 
        self.mask_for_overwrite[:] = self.default_mask[:]
        for (k, _, _) in individual:
            self.mask_for_overwrite[k] = False
            #self.distances_for_overwrite[k] = np.max(self.distances_for_overwrite)

        weights = self.sdrs_for_overwrite[self.mask_for_overwrite]
        probs =  weights / np.sum(weights)
        k = np.random.choice(self.node_indices_array[self.mask_for_overwrite], p = probs)
        psi = random.choice(self.problem.nodes[k].intervals).generate_uniform_angle()
        tau = (psi + np.pi + np.random.uniform( - self.eta / 2, self.eta / 2)) % (2 * np.pi)

        return (k, psi, tau)
                
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
    def fitness_function(self, individual: CEDOPADSRoute, ratio_of_time_used: float) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        # We wish to add a penalty to penalize routes which are to long
        # and to make the penalty more and more suvere as time is running out.
        length = self.problem.compute_length_of_route(individual, self.sensing_radius, self.rho)
        length_penalty = 1 / (1 + np.exp(-max(ratio_of_time_used * (length - self.t_max), 0)))

        score = self.problem.compute_score_of_route(individual, self.eta)
        return 2 * (1 - length_penalty) * score 
     
    # -------------------------------------- Recombination Operators -------------------------------------- #
    def k_point_crossover(self, parents: List[CEDOPADSRoute], k: int = 4) -> List[CEDOPADSRoute]:
        """Performs k-point crossover between the parents, and creates an offspring."""
        assert k >= 2 and len(parents) >= 2

        # Pick random inidices used to create the k-point crossovers, ie. each parent gets split at the 
        # indices and the subparts, gets rearanged back into a set of children, if this is not posible i.e.
        # if the length of the parent is less than k, simply remove it from the pool, and if we dont have
        # enough suffiicently long parents, we simply return an empty list.
        parents = [parent for parent in parents if len(parent) >= k]
        if len(parents) == 0:
            return []
        elif len(parents) == 1:
            return parents
        indices = [sorted(np.random.choice(np.arange(0, len(parent)), size = k, replace = False))
                    for parent in parents]
        
        # Generate all of the posible offspring from the given list of indices for each parent.
        offspring = []
        for parent_indices in combinations_with_replacement(range(len(parents)), k):
            
            # Generate the child which corresponds to the parent indices
            for jdx in parent_indices:
                parent = parents[jdx]
                for idx in range(k):
                    if idx == 0:
                        child = parent[:indices[jdx][0]]
                    elif idx == k - 1:
                        child.extend(parent[indices[jdx][k - 2]:])
                    else:
                        child.extend(parent[indices[jdx][idx]:indices[jdx][idx + 1]])

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

    def samir_crossover(self, parents: List[CEDOPADSRoute], k: int = 2) -> List[CEDOPADSRoute]:
        """Performs order crossover, by iteratively selecting two parents and producing their offspring."""
        # Compute offspring for each pair of parents included in the parents list.
        offspring = []
        for pool in combinations(parents, k):
            
            for i, dominant_parent in enumerate(pool):
                # Keep track of the current index in each of the parents.
                indices = {j: 0 if i != j else 1 for j in range(k)}
                child = [dominant_parent[0]]

                # Compute scores and distances.
                distances, scores = {}, {}
                for j, index in indices.items():
                    distances[j] = self.problem.compute_distance_between_visits(child[-1], pool[j][index], self.sensing_radius, self.rho)
                    scores[j] = self.problem.compute_score_of_visit(pool[j][index], self.eta) 
                
                # Perform a greedy choice of the next visit in the route based on the sdr score, 
                # from the final state in the route to the next candidate state.
                while problem.compute_length_of_route(child, self.sensing_radius, self.rho) < t_max:
                    sdr_scores_of_visits = [scores[j] / distances[j] for j in range(k)]

                    # Chose the next visit greedily, but make sure that we dont pick a node which
                    # has already been visited. TODO
                    greedy_idx = np.argmax(sdr_scores_of_visits)
                    child.append(pool[greedy_idx][indices[greedy_idx]])

                    # Move the pointer for the chosen route one step ahead, and update the scores and distances accordingly.
                    indices[greedy_idx] += 1
                    if indices[greedy_idx] >= len(pool[greedy_idx]):
                        # Simply pick a new random node for the 
                        pool[greedy_idx].append(self.pick_new_visit(pool[greedy_idx]))

                    scores[greedy_idx] = self.problem.compute_score_of_visit(pool[greedy_idx][indices[greedy_idx]], self.eta)

                    for j, index in indices.items():
                        distances[j] = self.problem.compute_distance_between_visits(child[-1], pool[j][indices[j]], self.sensing_radius, self.rho)

                    offspring.append(child[:-1])

        return offspring

    def single_index_greedy_crossover(self, parents: List[CEDOPADSRoute], k: int = 2) -> List[CEDOPADSRoute]:
        """Performs order crossover, by iteratively selecting two parents and producing their offspring."""
        # Compute offspring for each pair of parents included in the parents list.
        offspring = []
        for pool in combinations(parents, k):
            
            for i, dominant_parent in enumerate(pool):
                # Keep track of the current index in each of the parents.
                child = [dominant_parent[0]]
                index = 1
                seen = set() 

                # Perform a greedy choice of the next visit in the route based on the sdr score, 
                # from the final state in the route to the next candidate state.
                while self.problem.compute_length_of_route(child, self.sensing_radius, self.rho) < self.t_max:
                    q = self.problem.nodes[child[-1][0]].get_state(self.sensing_radius, child[-1][1], child[-1][2])
                    sdr_of_visits = []
                    for parent in pool:
                        # Make sure that every route in the pool has an appropriate length and that we have not already visited the greedily chosen node.
                        if index >= len(parent):
                            parent.append(self.pick_new_visit(child))
                        elif parent[index][0] in seen:
                            parent[index] = self.pick_new_visit(child)
                        
                        score = self.problem.compute_score_of_visit(parent[index], self.eta) 
                        distance = self.problem.compute_distance_between_state_and_visit(q, parent[index], self.sensing_radius, self.rho) 
                        sdr_of_visits.append(score / distance)


                    # Chose the next visit greedily, based on the sdr score.
                    i = np.argmax(sdr_of_visits)
                    child.append(pool[i][index])
                    seen.add(pool[i][index][0])
                    index += 1

                offspring.append(child[:-1])

    def multi_index_greedy_crossover_with_arbitrary_starting_point(self, parents: List[CEDOPADSRoute], scores: List[List[float]], states: List[List[State]], k: int = 2, n: int = 2) -> List[CEDOPADSRoute]:
        """Performs order crossover, by iteratively selecting two parents and producing their offspring."""
        # Compute offspring for each pair of parents included in the parents list.
        offspring = []

        number_of_candidates = k * n
        # NOTE: these repeatedly gets overwritten each time a new greedy choice is made.
        sdr_of_visits = np.zeros(number_of_candidates)
        distances = np.zeros(number_of_candidates)

        for indices in combinations(range(len(parents)), k):
            pool = [parents[index] for index in indices]
            
            for dominant_parent in pool:
                # Keep track of the current index in each of the parents.
                child = [dominant_parent[0]]
                pointer = 1
                seen = np.zeros(len(self.problem.nodes))
                seen[child[0][0]] = 1

                # Perform a greedy choice of the next visit in the route based on the sdr score, 
                # from the final state in the route to the next candidate state.
                q = self.problem.nodes[child[-1][0]].get_state(self.sensing_radius, child[-1][1], child[-1][2])
                total_distance = compute_length_of_relaxed_dubins_path(q.angle_complement() , self.problem.source, self.rho)
                while total_distance + compute_length_of_relaxed_dubins_path(q, self.problem.sink, self.rho) < t_max:
                    candidates = []

                    for i, (index, parent) in enumerate(zip(indices, pool)):
                        # Make sure that every route in the pool has an appropriate length and that we have not already visited the greedily chosen node.
                        for j, l in enumerate(range(pointer, pointer + n)):
                            if l >= len(parent) or seen[parent[l][0]] == 1: # TODO: Maybe this can chosen better using the sdr tensor.
                                new_visit = self.pick_new_visit(child)
                                distances[i * n + j] = self.problem.compute_distance_between_state_and_visit(q, new_visit, self.sensing_radius, self.rho) 
                                sdr_of_visits[i* n + j] = self.problem.compute_score_of_visit(new_visit, self.eta) / distances[i* n + j]
                                candidates.append(new_visit)

                            else:
                                candidates.append(parent[l])
                                distances[i * n + j] = dubins.shortest_path(q.to_tuple(), states[index][l].to_tuple(), self.rho).path_length()
                                sdr_of_visits[i * n + j] = scores[index][l] / distances[i * n + j]
                    
                    # Chose the next visit greedily, based on the sdr score.
                    # NOTE: Argmax seems to work alot better than chosing the next with a random probability,
                    # however this can be tested using c, since a higher c yields results more like if we had 
                    # used argmax to choose i.

                    probs = sdr_of_visits  / np.sum(sdr_of_visits)
                    i = np.random.choice(number_of_candidates, p = probs) 

                    #i = np.argmax(sdr_of_visits) 
                    child.append(candidates[i])
                    total_distance += distances[i]
                    seen[child[-1][0]] = 1
                    q = self.problem.nodes[child[-1][0]].get_state(self.sensing_radius, child[-1][1], child[-1][2])
                    pointer += 1

                offspring.append(child[:-1])

        return offspring

    def multi_index_greedy_crossover(self, parents: List[CEDOPADSRoute], scores: List[List[float]], states: List[List[State]], k: int = 2, n: int = 2) -> List[CEDOPADSRoute]:
        """Performs order crossover, by iteratively selecting two parents and producing their offspring."""
        # Compute offspring for each pair of parents included in the parents list.
        offspring = []

        number_of_candidates = k * n
        # NOTE: these repeatedly gets overwritten each time a new greedy choice is made.
        sdr_of_visits = np.zeros(number_of_candidates)
        distances = np.zeros(number_of_candidates)

        for indices in combinations(range(len(parents)), k):
            pool = [parents[index] for index in indices]
            
            for dominant_parent in pool:
                # Keep track of the current index in each of the parents.
                child = [dominant_parent[0]]
                pointer = 1
                seen = np.zeros(len(self.problem.nodes))
                seen[child[0][0]] = 1

                # Perform a greedy choice of the next visit in the route based on the sdr score, 
                # from the final state in the route to the next candidate state.
                q = self.problem.nodes[child[0][0]].get_state(self.sensing_radius, child[0][1], child[0][2])
                total_distance = compute_length_of_relaxed_dubins_path(q.angle_complement() , self.problem.source, self.rho)
                while total_distance + compute_length_of_relaxed_dubins_path(q, self.problem.sink, self.rho) < self.t_max:
                    candidates = []

                    for i, (index, parent) in enumerate(zip(indices, pool)):
                        # Make sure that every route in the pool has an appropriate length and that we have  already visited the greedily chosen node.
                        for j, l in enumerate(range(pointer, pointer + n)):
                            if l >= len(parent) or seen[parent[l][0]] == 1:
                                new_visit = self.pick_new_visit(child)
                                distances[i * n + j] = self.problem.compute_distance_between_state_and_visit(q, new_visit, self.sensing_radius, self.rho) 
                                sdr_of_visits[i * n + j] = self.problem.compute_score_of_visit(new_visit, self.eta) / distances[i * n + j]
                                candidates.append(new_visit)

                            else:
                                candidates.append(parent[l])
                                distances[i * n + j] = dubins.shortest_path(q.to_tuple(), states[index][l].to_tuple(), self.rho).path_length()
                                sdr_of_visits[i * n + j] = scores[index][l] / distances[i * n + j] 
                    
                    # Chose the next visit greedily, based on the sdr score.
                    # NOTE: Argmax seems to work alot better than chosing the next with a random probability,
                    # however this can be tested using c, since a higher c yields results more like if we had 
                    # used argmax to choose i.
                    probs = sdr_of_visits / np.sum(sdr_of_visits)
                    if np.isnan(probs).any(): 
                        # There has been some sort of error in the calculations and we have 
                        # no choice but to chose a random candidate, uniformly.
                        i = np.random.choice(number_of_candidates)

                    else:
                        i = np.random.choice(number_of_candidates, p = probs) 
                    

                    total_distance += distances[i]
                    child.append(candidates[i])

                    #i = np.argmax(sdr_of_visits) 
                    seen[child[-1][0]] = 1
                    q = self.problem.nodes[child[-1][0]].get_state(self.sensing_radius, child[-1][1], child[-1][2])
                    pointer += 1

                offspring.append(child[:-1])

        return offspring

    #--------------------------------------------- Mutation Operators -------------------------------------- #
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

        while np.random.uniform(0, 1) < p_s and len(individual) >= 2: 
            # Swap two random visists along the route described by the individual.
            indices = np.random.choice(len(individual), size = 2, replace = False)
            individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]

        return self.fix_length(self.angle_mutation(individual, q))

    # TODO: Mayor performance gains to be had here 
    def fix_length(self, individual: CEDOPADSRoute) -> CEDOPADSRoute: 
        """Removes visits from the individual until it obeys the distance constraint."""
        states = self.problem.get_states(individual, self.sensing_radius)
        tuples = [state.to_tuple() for state in states]
        
        # NOTE: These are stored in order to make it easier to look them up later.
        distance = self.problem.compute_length_of_route(individual, self.sensing_radius, self.rho)
        
        # Save the inverse of the SDR scores of each visit in the route. 
        #weights = [1 / self.sdr_tensor[self.number_of_nodes][individual[1][0]][individual[0][0]]]
        #for i in range(1, len(individual) - 1):
        #    weight = 1 / self.sdr_tensor[individual[i - 1][0]][individual[i + 1][0]][individual[i][0]]
        #    weights.append(weight)

        #weights.append(1 / self.sdr_tensor[self.number_of_nodes + 1][individual[-2][0]][individual[-1][0]])
        #sum_of_weights = sum(weights)


        # NOTE: There might be some errors in this function, since we could end up not being able to update the weights ect.
        while distance > self.t_max:
            # pick a stochastic index to remove based on the inverse of sdr

            weights = np.empty(len(individual))
            weights[0] = 1 / self.sdr_tensor[self.number_of_nodes][individual[1][0]][individual[0][0]]
            for i in range(1, len(individual) - 1):
                weights[i] = 1 / self.sdr_tensor[individual[i - 1][0]][individual[i + 1][0]][individual[i][0]]

            weights[-1] = 1 / self.sdr_tensor[self.number_of_nodes + 1][individual[-2][0]][individual[-1][0]]
            
            index = np.random.choice(len(individual), p = [weight / np.sum(weights) for weight in weights]) 
            
            # Update the distance and weights based on the index being poped.
            if index == 0:
                distance += (compute_length_of_relaxed_dubins_path(states[1].angle_complement(), self.problem.sink, self.rho) -
                             dubins.shortest_path(tuples[0], tuples[1], self.rho).path_length() -
                             compute_length_of_relaxed_dubins_path(states[0].angle_complement(), self.problem.sink, self.rho))
            elif index < len(individual) - 1:
                distance += (dubins.shortest_path(tuples[index - 1], tuples[index + 1], self.rho).path_length() -
                             dubins.shortest_path(tuples[index - 1], tuples[index], self.rho).path_length() -
                             dubins.shortest_path(tuples[index], tuples[index + 1], self.rho).path_length()) 
            else:
                distance += (compute_length_of_relaxed_dubins_path(states[index - 1], self.problem.sink, self.rho) -
                             dubins.shortest_path(tuples[index - 1], tuples[index], self.rho).path_length() -
                             compute_length_of_relaxed_dubins_path(states[index], self.problem.sink, self.rho))

            states.pop(index)
            tuples.pop(index)
            individual.pop(index)
        
        return individual

    def angle_mutation (self, individual: CEDOPADSRoute, q: float) -> CEDOPADSRoute:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
        # Core idea here is that in the start we are quite interested in making
        # large jumps in each psi and tau, however in the end we are more interested 
        # in wiggeling them around to perform the last bit of optimization
        for idx, (k, _, _) in enumerate(individual):
            u = random.uniform(0, 1)
            if u < q:
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
                new_tau = np.random.uniform(np.pi + new_psi - self.eta / 2, np.pi + new_psi + self.eta / 2) % (2 * np.pi)
                individual[idx] = (k, new_psi, new_tau)

                            # Generate a new tau non-uniformly based on the scale_for_tau argument
            # new_tau = truncnorm.rvs(-self.eta / (2 * scale_for_tau), self.eta / (2 * scale_for_tau), 
            #                         loc = np.pi + new_psi, scale = scale_for_tau) % (2 * np.pi)
            
        return individual

    # --------------------------------------- Local Search Operators ------------------------------------ #    
    def wiggle_angles(self, individual: CEDOPADSRoute, q: float, scale_modifier: float = 0.02, scale_for_tau: float = 0.3) -> CEDOPADSRoute:
        """Wiggles the angles of the individual slightly, using truncated normal distributions."""
        for idx, (k, psi, _) in enumerate(individual):
            if np.random.uniform(0, 1) < q:
                # Figure out which interval the current angle is located within.
                for jdx, interval in enumerate(self.problem.nodes[k].intervals):
                    if interval.contains(psi):
                        break

                # Set bounds for random number generation NOTE: There are checks for these in the data_types.py file.
                new_psi = interval.generate_truncated_normal_angle(mean = self.problem.nodes[k].thetas[jdx], scale_modifier = scale_modifier)
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

    def sigma_scaling(self, c: float = 2) -> ArrayLike:
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

    def roulette_wheel (self, cdf: ArrayLike, m: int) -> List[int]:
        """Implements roulette wheel sampling, to sample the indices of m parents from the population."""
        indices = []

        # Basically we just sample the population according to the probabilities given indirectly by the CDF
        # which corresponds to spinning m roulette wheels where the lengths of each segment is proportional to
        # the probability of picking each of the individuals of the population.
        for _ in range(m):
            r = np.random.uniform(0, 1)
            i = 0
            while r <= cdf[i] or i == self.mu - 1:
                i += 1
            
            indices.append(i - 1)
            
        return indices

    # ----------------------------------------- Survivor Selection -------------------------------------- #
    def mu_plus_lambda_selection(self, offspring: List[CEDOPADSRoute], ratio_of_time_used: float) -> List[CEDOPADSRoute]:
        """Merges the newly generated offspring with the generation and selects the best mu individuals to survive until the next generation."""
        # NOTE: Recall that the fitness is time dependent hence we need to recalculate it for the parents.
        fitnesses_of_parents_and_offspring = np.array(list(map(lambda child: self.fitness_function(child, ratio_of_time_used), self.population + offspring)))

        # Calculate the indices of the best performing memebers
        indices_of_new_generation = np.argsort(fitnesses_of_parents_and_offspring)[-self.mu:]
        
        # Simply set the new fitnesses which have been calculated recently.
        self.fitnesses = fitnesses_of_parents_and_offspring[indices_of_new_generation]

        return [self.population[i] if i < self.mu else offspring[i - self.mu] for i in indices_of_new_generation]

    def mu_comma_lambda_selection(self, offspring: List[CEDOPADSRoute], ratio_of_time_used: float) -> List[CEDOPADSRoute]:
        """Picks the mu offspring with the highest fitnesses to populate the the next generation, note this is done with elitism."""
        fitnesses_of_offspring = np.array(list(map(lambda child: self.fitness_function(child, ratio_of_time_used), offspring)))

        # Calculate the indices of the best performing memebers
        indices_of_new_generation = np.argsort(fitnesses_of_offspring)[-self.mu:]

        if fitnesses_of_offspring[indices_of_new_generation[0]] < self.highest_fitness_recorded:
            # Use elitism
            self.fitnesses = fitnesses_of_offspring[indices_of_new_generation[:-1]] + [self.highest_fitness_recorded]
            return [offspring[i] for i in indices_of_new_generation[:-1]] + [self.individual_with_highest_score]
        else:
            # Simply set the new fitnesses which have been calculated recently.
            self.fitnesses = fitnesses_of_offspring[indices_of_new_generation]
            return [offspring[i] for i in indices_of_new_generation]

    # -------------------------------------------- Main Loop of GA --------------------------------------- #
    def run(self, time_budget: float, m: int, parent_selection_mechanism: Callable[[], List[CEDOPADSRoute]], crossover_mechanism: Callable[[List[CEDOPADSRoute]], List[CEDOPADSRoute]], survivor_selection_mechanism: Callable[[List[CEDOPADSRoute], float], ArrayLike], p_c: float, progress_bar: bool = False) -> CEDOPADSRoute:
        """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
        assert p_c <= 1.0

        start_time = time.time()
        # Generate an initial population, which almost satifies the distant constraint, ie. add nodes until
        # the sink cannot be reached within d - t_max units where d denotes the length of the route.
        self.population = []
        for _ in range(self.mu):
            route = []
            while len(route) == 0 or self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) < self.t_max:
                route.append(self.pick_new_visit(route))

            self.population.append(route[:-1])

        # Compute fitnesses of each route in the population and associated probabilities using 
        # the parent_selection_mechanism passed to the method, which sould be one of the parent
        # selection methods defined earlier in the class declaration.
        self.fitnesses = [self.fitness_function(individual, 0) for individual in self.population]

        # NOTE: The fitnesses are set as a field of the class so they are accessable within the method
        probabilities = parent_selection_mechanism() 
        cdf = np.add.accumulate(probabilities)

        gen = 0
        best_route, self.highest_score = None, 0
        with tqdm(total=time_budget, desc="GA") as pbar:
            # Set initial progress bar information.
            idx_of_best_route = np.argmax(self.fitnesses)

            if progress_bar:
                avg_fitness = np.sum(self.fitnesses) / self.mu
                avg_distance = sum(self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) for route in self.population) / self.mu
                avg_length = sum(len(route) for route in self.population) / self.mu
                pbar.set_postfix({
                        "Gen": gen,
                        "Best": f"({self.fitnesses[idx_of_best_route]:.1f}, {self.problem.compute_length_of_route(self.population[idx_of_best_route], self.sensing_radius, self.rho):.1f}, {len(self.population[idx_of_best_route])})",
                        "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})"
                    })

            while (elapsed_time := (time.time() - start_time)) < time_budget:
                ratio_of_time_used = elapsed_time / time_budget

                # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
                # and perform k-point crossover using these parents, to create a total of m^k children which is
                # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
                # the angles in order to find new versions of the parents with higher fitness values.
                scores = [[self.problem.nodes[k].compute_score(psi, tau, self.eta) for (k, psi, tau) in parent] for parent in self.population]
                states = [[self.problem.nodes[k].get_state(self.sensing_radius, psi, tau) for (k, psi, tau) in parent] for parent in self.population]
                offspring = []
                for i in range(self.mu * p_c):
                    parent_indices = self.stochastic_universal_sampling(cdf, m = m)
                    
                    for child in crossover_mechanism([self.population[i] for i in parent_indices], [scores[i] for i in parent_indices], [states[i] for i in parent_indices]): 
                        mutated_child = self.mutate(child, p_s = 0.2, p_i = 0.3, p_r = 0.2, q = 0.1)
                        
                        # TODO: implement a local search improvement operator, ie. convert
                        # the algorithm to a Lamarckian memetatic algortihm

                        offspring.append(mutated_child) 
                        #offspring.append(child)
                    
                    # NOTE: we are simply interested in optimizing the existing angles within the individual,
                    # hence we create a new offspring where a few of the angles are different compared to 
                    # the original individual, sort of like a Lamarckian mematic algortihm, however
                    # here the local search is done using a stochastic operator.
                    offspring.append(self.fix_length(self.wiggle_angles(deepcopy(self.population[i]), q = np.power(ratio_of_time_used, 2) * 0.2)))

                # Replace the worst performing individuals based on their fitness values, 
                # however do it softly so that the population increases over time
                # Survivor selection mechanism (Replacement)
                gen += 1
                self.population = survivor_selection_mechanism(offspring, ratio_of_time_used)
                probabilities = parent_selection_mechanism() 
                cdf = np.add.accumulate(probabilities)


                idx_of_best_route = np.argmax(self.fitnesses)
                if (self.problem.compute_length_of_route(self.population[idx_of_best_route], self.sensing_radius, self.rho) < self.t_max and
                    self.fitnesses[idx_of_best_route] > self.highest_score):
                    best_route = self.population[idx_of_best_route]
                    self.highest_score = self.fitnesses[idx_of_best_route]

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                pbar.n = round(time.time() - start_time, 1)
                if progress_bar:
                    avg_fitness = np.mean(self.fitnesses)
                    avg_distance = sum(self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) for route in self.population) / self.mu
                    avg_length = sum(len(route) for route in self.population) / self.mu
                    pbar.set_postfix({
                            "Gen": gen,
                            "Best": f"({self.fitnesses[idx_of_best_route]:.1f}, {self.problem.compute_length_of_route(self.population[idx_of_best_route], self.sensing_radius, self.rho):.1f}, {len(self.population[idx_of_best_route])})",
                            "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})"
                    })
    
        # TODO: Finally try to fix routes which are to long before returning the best route
        permitable_routes = list(filter(lambda route: self.problem.compute_length_of_route(route, self.sensing_radius, self.rho) < self.t_max, self.population))
        best_permitable_route = max(permitable_routes, key = lambda route: self.problem.compute_score_of_route(route, self.eta))
        if self.highest_score < self.problem.compute_score_of_route(best_permitable_route, self.eta):
            return best_permitable_route
        else:
            return best_route

    # ------------------------------------------------ MISC --------------------------------------------- #
    def __repr__(self) -> str:
        """Returns a string representation of the information stored in the algorithm."""
        return f"t_max: {self.t_max:.1f}, sensing_radius: {self.sensing_radius:.1f}, rho: {self.rho:.1f}, eta: {self.eta:.1f}, mu: {self.mu}"

if __name__ == "__main__":
    problem: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
    t_max = 200 
    rho = 2
    sensing_radius = 2
    eta = np.pi / 5
    ga = GA(problem, t_max, eta, rho, sensing_radius, mu = 2048)
    print(ga)
    #cProfile.run("ga.run(60, 3, ga.sigma_scaling, ga.multi_index_greedy_crossover, ga.mu_comma_lambda_selection)", sort = "cumtime")
    route = ga.run(300, 2, ga.sigma_scaling, ga.multi_index_greedy_crossover, ga.mu_comma_lambda_selection)
    problem.plot_with_route(route, sensing_radius, rho, eta)
    plt.show()
