# %% 
import time
from typing import Tuple, Set, Callable, Iterator, List, Optional, Dict

import matplotlib.pyplot as plt 
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSNode, load_CEDOPADS_instances, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from library.CEDOPADS_solvers.local_search_operators import add_free_visits, hill_climbing, get_samples
from classes.data_types import Angle, State
import dubins
from library.CEDOPADS_solvers.greedy import greedy 
from dataclasses import dataclass

@dataclass
class CEDOPADSRouteInfo:
    indices: List[Tuple[int, int]] # k & i which can be used to get the states by  
    scores: List[float]
    lengths: List[float]

class GRASP:
    distances_from_source: Dict[Tuple[int, int], float]
    distances_to_sink: Dict[Tuple[int, int], float]

    distances: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] # distances[(k0, i0), (k1, i1)] is the length of the shortest dubins path from state[k0][i0] to state[k1][i1]

    def __init__ (self, problem: CEDOPADSInstance, utility_function: UtilityFunction):
        """Initializes the GRASP algorithm"""
        self.problem_instance = problem
        self.utility_function = utility_function

    def greedy(self, p = 1.0) -> CEDOPADSRouteInfo:
        """Greedily constructs a CEDOPADS route"""
        remaining_distance = self.problem_instance.t_max
        indices = []
        scores = []
        lengths = []

        # 1. Pick first visit 
        candidate_indices = {}
        unvisited_nodes = {k for k in range(len(self.problem_instance.nodes))}
        for k in unvisited_nodes:
            for i, score in enumerate(self.scores[k]):
                candidate_indices[(k, i)] = score / self.distances_from_source[(k, i)]

        # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
        if p < 1.0:
            N = int(np.ceil(len(candidate_indices) * (1 - p)))
        else:
            N = 1

        rcl = [tup[0] for tup in sorted(candidate_indices.items(), key = lambda tup: tup[1], reverse=True)[:N]] 
        k0, i0 = rcl[np.random.randint(0, len(rcl))]
        unvisited_nodes.remove(k0)
        indices.append((k0, i0))
        scores.append(self.scores[k0][i0])
        lengths.append(self.distances_from_source[(k0, i0)])

        # Compute the remaning distance.
        remaining_distance -= lengths[-1] 

        # 2. Keep going until we reach the sink, in each of the routes.
        while True:
            candidate_indices = {}
            q0 = self.states[k0][i0].to_tuple()
            for k in unvisited_nodes:
                for i, score in enumerate(self.scores[k]):
                    key = ((k0, i0), (k, i))
                    # Compute the distance from self.states[k0][i0] if needed
                    if not (key in self.distances.keys()):
                        self.distances[key] = dubins.shortest_path(q0, self.states[k][i].to_tuple(), self.problem_instance.rho).path_length() 

                    length = self.distances[key]

                    if self.distances_to_sink[(k, i)] + length <= remaining_distance:
                        candidate_indices[(k, i)] = score / length

            if len(candidate_indices) == 0:
                break

            # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
            if p < 1.0:
                N = int(np.ceil(len(candidate_indices) * (1 - p)))
            else:
                N = 1

            rcl = [tup[0] for tup in sorted(candidate_indices.items(), key = lambda tup: tup[1], reverse=True)[:N]] 
            k, i = rcl[np.random.randint(0, len(rcl))]
            unvisited_nodes.remove(k)
            indices.append((k, i))
            scores.append(self.scores[k][i])
            lengths.append(self.distances[((k0, i0), (k, i))])
            k0, i0 = k, i

            # Compute the remaning distance.
            remaining_distance -= lengths[-1] 

        lengths.append(self.distances_to_sink[indices[-1]])

        return CEDOPADSRouteInfo(indices, scores, lengths)

    def remove_visit (self, info: CEDOPADSRouteInfo) -> Tuple[CEDOPADSRouteInfo, float, float]:
        """Remove the worst performing node (measured via the SDR) from the route."""
        if len(info.indices) == 1:
            new_length = np.linalg.norm(self.problem_instance.source - self.problem_instance.sink)
            return (CEDOPADSRouteInfo([], [], [new_length]), new_length - sum(info.lengths), -info.scores[0])
            
        # Contains the new lengths if the visit at index idx is skipped.
        new_lengths_from_skipping = [self.distances_from_source[info.indices[1]]]
        for idx in range(1, len(info.indices) - 1):
            k0, i0 = info.indices[idx - 1]
            k1, i1 = info.indices[idx + 1]

            if not (((k0, i0), (k1, i1)) in self.distances.keys()):
                # Compute distances.
                q0 = self.states[k0][i0].to_tuple()
                self.distances[((k0, i0), (k1, i1))] = {i: dubins.shortest_path(q0, q.to_tuple(), self.problem_instance.rho).path_length() for i, q in enumerate(self.states[k1])}
 
            new_lengths_from_skipping.append(self.distances[((k0, i0), (k1, i1))])

        new_lengths_from_skipping.append(self.distances_to_sink[info.indices[-2]])

        # Find the index of the lowest sdr score. i.e. the one where we save the most distance for the least penality in score
        sdr_scores = [(score / ((length_before + length_after) - new_length_from_skipping)) for (score, new_length_from_skipping, length_before, length_after) in zip(info.scores, new_lengths_from_skipping, info.lengths[:-1], info.lengths[1:])]

        idx = np.argmin(sdr_scores)
        
        # Compute the saved distance & lost score
        change_in_length = new_lengths_from_skipping[idx] - (info.lengths[idx] + info.lengths[idx + 1]) 
        change_in_score = -info.scores[idx]

        # Update indices ect.
        new_indices = info.indices[:idx] + info.indices[idx + 1:]
        new_scores = info.scores[:idx] + info.scores[idx + 1:]

        if idx == 0:
            new_lengths = [new_lengths_from_skipping[0]] + info.lengths[2:]
        elif idx == len(info.indices): # Remember this is one less than it was before.
            new_lengths = info.lengths[:-2] + [new_lengths_from_skipping[-1]]
        else:
            new_lengths = info.lengths[:idx] + [new_lengths_from_skipping[idx]] + info.lengths[idx + 2:]

        return (CEDOPADSRouteInfo(new_indices, new_scores, new_lengths), change_in_length, change_in_score)

    def add_visit (self, info: CEDOPADSRouteInfo) -> bool:
        """Greedily add a visit to the route, if posible, otherwise return None"""
        remaining_distance = self.problem_instance.t_max - sum(info.lengths)

        candiate_nodes = {k for k in range(len(self.problem_instance.nodes))}.difference([k for k, _ in info.indices])

        index_for_insertion = None
        best_indices: Tuple[int, int] = None
        best_sdr = -np.inf
        best_lengths: Tuple[float, float] = (0, 0)

        # Check if the new visit should be inserted before the first visit
        q0 = self.states[info.indices[0][0]][info.indices[0][1]].to_tuple()
        for k in candiate_nodes:
            # Compute distances from the visit that we are considering to info.states[0] in case they have not already been computed.
            for i in range(len(self.visits[k])):
                key = ((k, i), info.indices[0])
                if not (key in self.distances.keys()):
                   self.distances[key] = dubins.shortest_path(self.states[k][i].to_tuple(), q0, self.problem_instance.rho).path_length()

                change_in_length = (self.distances[key] + self.distances_from_source[(k, i)]) - info.lengths[0]
                if change_in_length <= remaining_distance and (sdr := self.scores[k][i] / change_in_length) > best_sdr:
                    best_sdr = sdr
                    best_lengths = (self.distances_from_source[(k, i)], self.distances[key])
                    best_indices = (k, i)
                    index_for_insertion = 0               
               
        # Check if the visit should be inserted in the middle of the route.
        for j, (idxs, jdxs) in enumerate(zip(info.indices[:-1], info.indices[1:])):
            previous_length = info.lengths[j + 1] # NOTE: The first length is the length from the source to the first visit.
            q0, q1 = self.states[idxs[0]][idxs[1]].to_tuple(), self.states[jdxs[0]][jdxs[1]].to_tuple()
            for k in candiate_nodes:
                for i in range(len(self.visits[k])):
                    key0, key1 = (idxs, (k, i)), ((k, i), jdxs)
                    q = self.states[k][i].to_tuple()
                    if not (key0 in self.distances.keys()):
                        self.distances[key0] = dubins.shortest_path(q0, q, self.problem_instance.rho).path_length()
                    
                    if not (key1 in self.distances.keys()):
                        self.distances[key1] = dubins.shortest_path(q, q1, self.problem_instance.rho).path_length()
                    
                    change_in_length = (self.distances[key0] + self.distances[key1]) - previous_length
                    
                    if change_in_length <= remaining_distance and (sdr := self.scores[k][i] / change_in_length) > best_sdr:
                        best_sdr = sdr
                        best_lengths = (self.distances[key0], self.distances[key1])
                        best_indices = (k, i)
                        index_for_insertion = j + 1
               
        # Check if the visit should be insert at the end of the route.
        q0 = self.states[info.indices[-1][0]][info.indices[-1][1]].to_tuple()
        for k in candiate_nodes:
            # Compute distances from the visit that we are considering to info.states[0] in case they have not already been computed.
            for i in range(len(self.visits[k])):
                key = (info.indices[-1], (k, i))
                if not (key in self.distances.keys()):
                   self.distances[key] = dubins.shortest_path(q0, self.states[k][i].to_tuple(), self.problem_instance.rho).path_length()

                change_in_length = (self.distances[key] + self.distances_to_sink[(k, i)]) - info.lengths[-1]
                if change_in_length <= remaining_distance and (sdr := self.scores[k][i] / change_in_length) > best_sdr:
                    best_sdr = sdr
                    best_lengths = (self.distances[key], self.distances_to_sink[(k, i)])
                    best_indices = (k, i)
                    index_for_insertion = len(info.indices) 

        # We did not find any visits which could be included into the route.
        if best_indices is None:
            return False
        
        # Do the actual insertion, into the route 
        info.indices.insert(index_for_insertion, best_indices)       
        info.scores.insert(index_for_insertion, self.scores[best_indices[0]][best_indices[1]])

        if index_for_insertion == 0:
            info.lengths.insert(index_for_insertion, best_lengths[0])
            info.lengths[1] = best_lengths[1]
        elif index_for_insertion == len(info.indices):
            info.lengths.insert(index_for_insertion, best_lengths[1])
            info.lengths[-2] = best_lengths[0]
        else:
            info.lengths.insert(index_for_insertion, best_lengths[0])
            info.lengths[index_for_insertion + 1] = best_lengths[1]

        return True 
        
    def hill_climb(self, info: CEDOPADSRouteInfo) -> CEDOPADSRouteInfo:
        """Perform hill climbing to improve the route"""
        # We will never add the same nodes to two distinct routes.
        best_info = info
        best_score = sum(info.scores)
        best_distance = sum(info.lengths)

        # Remove nodes while we increase the SDR metric of the route.
        while True:
            info, change_in_distance, change_in_score = self.remove_visit(best_info)
            if (best_score + change_in_score) / (best_distance + change_in_distance) > best_score / best_distance:
                best_info = info
                best_score += change_in_score
                best_distance += change_in_distance
            else: 
                break

        # Add as many nodes as posible to the route, in order to increase the score.
        # until no more nodes can be added in which case the add_visit method returns false.
        while self.add_visit(best_info): 
            continue
            
        return best_info 

    def convert_info_to_route(self, info: CEDOPADSRouteInfo) -> CEDOPADSRoute:
        """Converts an info object to an actual route."""
        return [self.visits[k][i] for k, i in info.indices]

    def run(self, time_budget: float, p: float, sampler: Callable[[CEDOPADSInstance, int], Iterator[Tuple[Angle, Angle]]], verbose: bool = False) -> CEDOPADSRoute:
        """Runs the GRASP algorithm upon the problem instance to obtain a route."""
        r = sum(self.problem_instance.sensing_radii) / 2
        start_time = time.time()

        # Compute the states, visits and scores
        self.visits = {}
        self.states = {} 
        self.scores = {}

        # Is used to store the distance of a dubins vehicle from a state to another so self.distances[((k, i), (k', i'))] 
        # will contain the distance from self.states[k][i] to self.states[k'][i']
        self.distances = {}

        # self.distances_to_sink[(k, i)] = distance from self.states[k][i] to the sink
        # self.distances_from_source[(k, i)] = distance source to self.states[k][i]
        self.distances_to_sink = {}
        self.distances_from_source = {}
        
        for k in range(len(self.problem_instance.nodes)):
            self.visits[k] = [(k, psi, tau, r) for (psi, tau) in sampler(self.problem_instance, k)]
            self.states[k] = self.problem_instance.get_states(self.visits[k])
            self.scores[k] = self.problem_instance.compute_scores_along_route(self.visits[k], self.utility_function)

            # Precomoute lengths to source and sink.
            for i, q in enumerate(self.states[k]):
                self.distances_to_sink[(k, i)] = compute_length_of_relaxed_dubins_path(q, self.problem_instance.sink, self.problem_instance.rho)
                self.distances_from_source[(k, i)] = compute_length_of_relaxed_dubins_path(q.angle_complement(), self.problem_instance.source, self.problem_instance.rho) 
        
        best_route_info = self.greedy()
        best_score = sum(best_route_info.scores) 
        number_of_routes = 1 # Keep track of number of routes t ocheck performance.

        while (time.time() - start_time < time_budget):
            candidate_route_info = self.greedy(p = p)
            candidate_route_info = self.hill_climb(candidate_route_info)

            if (score := sum(candidate_route_info.scores)) > best_score:
                best_route = self.convert_info_to_route(candidate_route_info)
                if verbose:
                    print(f"Information on best route: score={self.problem_instance.compute_score_of_route(best_route, self.utility_function)} length={self.problem_instance.compute_length_of_route(best_route)}")
                best_route_info = candidate_route_info
                best_score = score

            number_of_routes += 1

        print(f"{number_of_routes=}")
        return self.convert_info_to_route(best_route_info)

def main():
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.g.c.a.txt", needs_plotting = True)
    grasp = GRASP(problem_instance, utility_function)
    route = grasp.run(300, 0.99, get_samples)
    problem_instance.plot_with_route(route, utility_function)
    plt.show()
    #print(problem_instance.compute_score_of_route(add_free_visits(problem_instance, route, utility_function), utility_function))

if __name__ == "__main__":
    main()
    