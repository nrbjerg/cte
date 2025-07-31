import dubins 
import random 
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
from numpy.typing import ArrayLike
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, Visit, UtilityFunction, utility_fixed_optical
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path, compute_relaxed_dubins_path, CSPath, CCPath
from typing import List, Tuple, Iterator, Optional
from classes.data_types import State, Matrix, Vector, compute_difference_between_angles, Dir
from library.core.dubins.dubins_api import call_dubins, compute_minimum_distance_to_point_from_dubins_path_segment
from library.CEDOPADS_solvers.local_search_operators import add_free_visits 
import hashlib
import itertools
from collections import Counter

@dataclass
class Individual:
	"""Models a invidudial in the memetic algorithm."""
	route: CEDOPADSRoute
	aoa_indices: List[int]

	scores: List[float] = None
	states: List[State] = None
	segment_lengths: List[float] = None

	def __len__ (self):
		"""Returns the number of visits in the route."""
		return len(self.route)
    
	def __repr__ (self) -> str:
		"""Returns a string representation of the individual."""
		visits = [f"({int(k)}, {float(psi):.1f}, {float(tau):.1f}, {r:.1f})" for (k, psi, tau, r) in self.route]
		return "[" + " ".join(visits) + "]"

class MemeticAlgorithm:
	"""Memetic algorithm for solving CEDOPADS problems."""
	problem_instance: CEDOPADSInstance
	population: List[Individual]
	utility_function: UtilityFunction
	fitnesses: Vector
 
	# Parameters for the memetic algorithm, controlling the population size and number of offspring.
	mu: int
	lmbda: int
	
	# Parameters for operators.
	M: int

    # Parameters which makes sure that we dont have to recompute the same values over and over again.
	number_of_nodes: int
	node_indices_array: Vector

	def __init__(self, problem_instance: CEDOPADSInstance, M: int = 3, mu: int = 100, lmbda: int = 100, utility_function: UtilityFunction = utility_fixed_optical, seed: Optional[int] = None):
		"""Initializes the memetic algorithm with a given problem instance and parameters."""
		self.problem_instance = problem_instance
		self.M = M
		self.mu = mu
		self.lmbda = lmbda
		self.utility_function = utility_function

		self.number_of_nodes = len(problem_instance.nodes)
		self.node_indices_array = np.array(range(self.number_of_nodes))

		if seed is None:
			hash_of_id = hashlib.sha1(problem_instance.problem_id.encode("utf-8")).hexdigest()[:8]
			seed = int(hash_of_id, 16)
        
		np.random.seed(seed)
		random.seed(seed)
        
		# Make sure that the sink can be reached from the source otherwise there is no point in running the algorithm.
		if (d := np.linalg.norm(problem_instance.source - problem_instance.sink)) > problem_instance.t_max:
			raise ValueError(f"Cannot construct a valid Dubins route since ||source - sink|| = {d} is greater than t_max = {problem_instance.t_max}")

	def initialize_population(self) -> List[CEDOPADSRoute]:	
		"""Initializes the population with random routes."""
		node_indices = set(range(self.number_of_nodes))
		population = []
		for _ in range(self.mu):
			route = [] 
			blacklist = set()

			aoa_indices = []
			while self.problem_instance.is_route_feasible(route):
				# Pick a new random visit and add it to the route.
				k = random.choice(list(node_indices.difference(blacklist)))
				blacklist.add(k)
				aoa_indices.append(np.random.randint(len(self.problem_instance.nodes[k].AOAs)))
				psi = self.problem_instance.nodes[k].AOAs[aoa_indices[-1]].generate_uniform_angle() 
				tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
				r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

				route.append((k, psi, tau, r)) 

            # Initialize what we need to know from the individual
			scores = self.problem_instance.compute_scores_along_route(route, self.utility_function)

		population.append(Individual(route, scores = scores,
									 aoa_indices = aoa_indices,
									 states = self.problem_instance.get_states(route))) 

		return population

	# ------------------------------------------------------- Fitness -------------------------------------------------------
	def fitness_function(self, individual: Individual) -> float:
		"""Computes the fitness of an individual."""
		if individual.scores is None:
			individual.scores = self.problem_instance.compute_scores_along_route(individual.route, self.utility_function)

		return sum(individual.scores)

	# ------------------------------------------------ Recombination Operators -----------------------------------------------------
	def modified_partially_mapped_crossover(self, parents: List[Individual]) -> List[Individual]:
		"""A modified version of the partially mapped crossover (PMX) operation on the parents to generate two offspring,
    	   which checks to find the possible crossover points where the parents visit the same node."""
		m = min(len(parents) for parent in parents)
		if m < 2:
			# In this case we cannot perform a crossover, so we just return the parents.
			return parents

		# Find candidate crossover point indices (i0, j0) in {0,..., len(parents[0]) - 1} and (i1, j1) in {0,...,len(parents[1]) - 1}.
		common_nodes = set.intersection(*(set(visit[0] for visit in parent.route) for parent in parents))

		if len(common_nodes) == 0:
			# No nodes in common, so we simply pick two random points within the parents.
			(i0, j0) = sorted(np.random.choice(len(parents[0]), 2, replace = False))
			(i1, j1) = sorted(np.random.choice(len(parents[1]), 2, replace = False))

		elif len(common_nodes) == 1:
			common_node_idx = common_nodes.pop()

			# Find the index of the common node in each parent.
			for i, (k, _, _, _) in enumerate(parents[0].route):
				if k == common_node_idx:
					i0 = i
					break
			
			for i, (k, _, _, _) in enumerate(parents[1].route):
				if k == common_node_idx:
					i1 = i
					break
			
			# Now pick a random other crossover points within each of the parents.
			j0 = np.random.choice([i for i in range(len(parents[0])) if i != i0], replace = False)
			j1 = np.random.choice([i for i in range(len(parents[1])) if i != i1], replace = False)
         
			(i0, j0) = sorted((i0, j0))
			(i1, j1) = sorted((i1, j1))
		
		else:
			# Pick two common nodes at random.
			common_nodes = np.random.choice(list(common_nodes), 2, replace=False)

			(i0, j0) = (None, None)
			# Find the index of the common nodes in each parent.
			for i, (k, _, _, _) in enumerate(parents[0].route):
				if k in common_nodes:
					if i0 is None:
						i0 = i
					else:
						j0 = i
						break

			(i1, j1) = (None, None)
			for i, (k, _, _, _) in enumerate(parents[1].route):
				if k in common_nodes:
					if i1 is None:
						i1 = i
					else:
						j1 = i
						break
			
		# Now we have the crossover points, we can perform the crossover.
		offspring = []
		for (paternal, (ip, jp)), (maternal, (im, jm)) in itertools.permutations([(parents[0], (i0, j0)), (parents[1], (i1, j1))], 2):
			route = paternal.route[:ip] + maternal.route[im:jm] + paternal.route[jp:]
			aoa_indices = paternal.aoa_indices[:ip] + maternal.aoa_indices[im:jm] + paternal.aoa_indices[jp:]
			states = paternal.states[:ip] + maternal.states[im:jm] + paternal.states[jp:]
			scores = paternal.scores[:ip] + maternal.scores[im:jm] + paternal.scores[jp:]

			# Fix any mishaps in the route, such as duplicate visits to the same node.
			ks = {k for (k, _, _, _) in maternal.route[im:jm]}
			possible_replacements = [idx for idx in range(ip, jp) if paternal.route[idx][0] not in ks]

			indices_to_remove = []
			visits_replaced = 0
			for idx in itertools.chain(range(ip), range(ip + (jm - im), len(route))):
				if route[idx][0] in ks:
					# We have a duplicate visit, so we need to replace it with a visit to a node that is not already in the route.
					if visits_replaced >= len(possible_replacements):
						# No possible replacements, so we just skip this individual.
						indices_to_remove.append(idx)
						
					else:
						replacement_idx = possible_replacements.pop(0)
						route[idx] = paternal.route[replacement_idx]
						aoa_indices[idx] = paternal.aoa_indices[replacement_idx]
						states[idx] = paternal.states[replacement_idx]
						scores[idx] = paternal.scores[replacement_idx]
						visits_replaced += 1
			
			# Remove any indices that were marked for removal.
			for idx in reversed(indices_to_remove):
				route.pop(idx)
				aoa_indices.pop(idx)
				states.pop(idx)
				scores.pop(idx)
   
			assert Counter(k for (k, _, _, _) in route) == Counter({k for (k, _, _, _) in route})

			offspring.append(Individual(route, aoa_indices=aoa_indices, states=states, scores=scores))
		
		return offspring

	def pick_visit_to_insert_based_on_SDR(self, route: CEDOPADSRoute, i: int) -> Tuple[Visit, float]:
		"""Picks the visit to insert at index i based on their SDR scores."""
		# TODO: Pick k based on distance to the path segment between the visits at i - 1 and i.
		# Its okay if everything else stays stochastic for the moment.
		if len(route) == 0:
			blacklist = set(visit[0] for visit in route)
			candidates = list(set(range(self.number_of_nodes)).difference(blacklist))
			k = np.random.choice(candidates)
			#k = random.choice(self.number_of_nodes)
		
		else:
			blacklist = set(visit[0] for visit in route)
			candidates = list(set(range(self.number_of_nodes)).difference(blacklist))
			
			# Pick k based on distance to the path segment between the visits at i - 1 and i if i != 0 and i != len(route) - 1.
			# otherwise look at the distance between the source and visit i or the distance between visit i - 1 and the sink.
			if i != 0 and i != len(route):
				q_i, q_f = self.problem_instance.get_states(route[i - 1:i + 1])
				(sub_segment_types, sub_segment_lengths) = call_dubins(q_i, q_f, self.problem_instance.rho)

			else:
				q_i = self.problem_instance.get_state(route[i]).angle_complement() if i == 0 else self.problem_instance.get_state(route[-1])
				path_segment = compute_relaxed_dubins_path(q_i, self.problem_instance.source, self.problem_instance.rho)
				
				# Check the two types.
				if type(path_segment) == CSPath:
					sub_segment_types = [path_segment.direction, Dir.S]
					length_of_initial_subsegment = self.problem_instance.rho * path_segment.radians_traversed_on_arc
					sub_segment_lengths = [length_of_initial_subsegment, path_segment.length - length_of_initial_subsegment]

				elif type(path_segment) == CCPath:
					sub_segment_types = path_segment.directions
					sub_segment_lengths = [self.problem_instance.rho * radians for radians in path_segment.radians_traversed_on_arcs]

			distances = [compute_minimum_distance_to_point_from_dubins_path_segment(self.problem_instance.nodes[k].pos, q_i, sub_segment_types, sub_segment_lengths, self.problem_instance.rho) for k in candidates]

			# If the minimum distance to the points are less than the sensing radius we dont care, but now we have the distances, we can compute the SDR scores.
			sdr_scores = np.array([self.problem_instance.nodes[k].base_line_score / max(distance, self.problem_instance.sensing_radii[1]) for k, distance in zip(candidates, distances)])
			k = candidates[np.random.choice(len(candidates), p = sdr_scores / np.sum(sdr_scores))]
		
		# After a node index k has been picked select the best of self.M visits to insert at index i based on SDR values.
		visits, sdr_scores = [], []
		for _ in range(self.M):
			aoa_indices = np.random.randint(len(self.problem_instance.nodes[k].AOAs))
			psi = self.problem_instance.nodes[k].AOAs[aoa_indices].generate_uniform_angle()
			tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
			r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])
			visits.append((k, psi, tau, r))
			score = self.problem_instance.compute_score_of_visit((k, psi, tau, r), self.utility_function)

			# Compute SDR score based on the new visit and the visit before and or after it in the route.
			q = self.problem_instance.get_state(visits[-1])
			if len(route) != 0:
				# Compute distance through the visit if it is inserted into the route at index i.
				if i == 0:
					distance_through_visit = (compute_length_of_relaxed_dubins_path(q.angle_complement(), self.problem_instance.source, self.problem_instance.rho) +
                               				  dubins.shortest_path(q.angle_complement().to_tuple(), self.problem_instance.get_state(route[0]).to_tuple(), self.problem_instance.rho).path_length())
				elif i == len(route):
					distance_through_visit = (dubins.shortest_path(self.problem_instance.get_state(route[-1]).to_tuple(), q.to_tuple(), self.problem_instance.rho).path_length() +
							   				  compute_length_of_relaxed_dubins_path(q, self.problem_instance.sink, self.problem_instance.rho))
				else: 
					distance_through_visit = (dubins.shortest_path(self.problem_instance.get_state(route[i - 1]).to_tuple(), q.to_tuple(), self.problem_instance.rho).path_length() +
							   				  dubins.shortest_path(q.to_tuple(), self.problem_instance.get_state(route[i]).to_tuple(), self.problem_instance.rho).path_length())	

				sdr_scores.append(score / distance_through_visit)

			else:
				# If the route is empty, we cannot compute the SDR score, so we just append the score.
				sdr_scores.append(score / self.problem_instance.compute_length_of_route([visits[-1]]))
		
		visit = visits[np.argmax(sdr_scores)]

		return (k, psi, tau, r), self.problem_instance.compute_score_of_visit((k, psi, tau, r), self.utility_function)

	# ------------------------------------------------------- Mutation Operators -------------------------------------------------------
	def mutate(self, individual: Individual, p: float, q: float, relative_sigma: float) -> Individual:
		"""Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route.""" 
		individual.segment_lengths = None
		# Replace visits within route
		n = max(0, min(np.random.geometric(1 - p) - 1, len(individual) - 1))
		indices = sorted(np.random.choice(len(individual), n, replace = False))
		for i in indices:
			# Replace visits through route.
			route_without_visit_i = individual.route[:i] + individual.route[i + 1:]
			best_visit, score = self.pick_visit_to_insert_based_on_SDR(route_without_visit_i, i)

			# Actually replace the visit in the individual
			individual.route[i] = best_visit
			aoa = self.problem_instance.nodes[best_visit[0]].get_AOA(best_visit[1])
			individual.aoa_indices[i] = self.problem_instance.nodes[best_visit[0]].AOAs.index(aoa)

			individual.states[i] = self.problem_instance.get_state(best_visit)
			individual.scores[i] = score

		indices_to_skip = indices

		# Insert visits within the route the route.
		n = max(0, min(np.random.geometric(1 - q) - 1, len(individual)))
		indices = sorted(np.random.choice(len(individual) + 1, n, replace = False))
		for j, i in enumerate(reversed(indices), 1):
			best_visit, score = self.pick_visit_to_insert_based_on_SDR(individual.route, i)

			# Actually insert the information into the individual
			individual.route.insert(i, best_visit)
			individual.scores.insert(i, score)
			individual.states.insert(i, self.problem_instance.get_state(best_visit))

			aoa = self.problem_instance.nodes[best_visit[0]].get_AOA(best_visit[1])
			individual.aoa_indices.insert(i, self.problem_instance.nodes[best_visit[0]].AOAs.index(aoa))
			indices_to_skip.append(i + n - j)

		# Finally we perform multiple angle mutations
		return self.mutate_scalars(individual, relative_sigma), indices_to_skip
	
	def mutate_scalars(self, individual: Individual, relative_sigma: float) -> Individual:
		"""Mutates the scalar values of the individual, i.e. the psi, tau and r values of the visits."""
		# For now simply pick a random visit and mutate it.
		if len(individual) == 0:	
			return individual
		
		# Pick a random visit to mutate.
		for i, ((k, psi, tau, r), aoa_idx) in enumerate(zip(individual.route, individual.aoa_indices)):
			aoa = self.problem_instance.nodes[k].AOAs[aoa_idx]
			phi = aoa.phi
			N_psi = np.random.normal(0, relative_sigma * phi)
			new_psi = psi + N_psi

			if not aoa.contains_angle(new_psi):
				change_in_aoa_idx = 1 if N_psi > 0 else -1 
				individual.aoa_indices[i] = (individual.aoa_indices[i] + change_in_aoa_idx) % len(self.problem_instance.nodes[k].AOAs) 
				aoa = self.problem_instance.nodes[k].AOAs[individual.aoa_indices[i]]
				new_psi = aoa.generate_uniform_angle() 
				delta_psi = new_psi - psi

			else:
				delta_psi = N_psi

			new_tau = (tau + np.random.normal(0, relative_sigma * self.problem_instance.eta / 2) + delta_psi + np.pi) % (2 * np.pi)
			if compute_difference_between_angles(new_tau, (new_psi + np.pi) % (2 * np.pi)) > self.problem_instance.eta / 2:
                # If this is not the case, simply move tau the same amount as N_tau.
				new_tau = (tau + delta_psi) % (2 * np.pi) 
   
			new_r = np.clip(r + np.random.normal(0, relative_sigma * (self.problem_instance.sensing_radii[1] - self.problem_instance.sensing_radii[0]) / 2),
				self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

			individual.route[i] = (k, new_psi, new_tau, new_r)
			individual.states[i] = self.problem_instance.get_state(individual.route[i])
			individual.scores[i] = self.problem_instance.compute_score_of_visit(individual.route[i], self.utility_function)

		return individual
	
	# ----------------------------------------- Local search operators -------------------------------------------------------
	def fix_length(self, individual: Individual, indices_to_skip: List[int]) -> Individual:
		"""Removes visits from the individual until the length of the route is less than or equal to t_max."""
		# Compute the length of the route segments if they have not already been precomputed.
		if individual.segment_lengths is None:
			individual.segment_lengths = self.problem_instance.compute_lengths_of_route_segments_from_states(individual.states)
		
		# Keep track of the index of the visit that we want to skip, this is the one with the lowest SDR score.
		idx_skipped = None

		total_distance = sum(individual.segment_lengths)
		while total_distance > self.problem_instance.t_max:
			if idx_skipped is None:
				# Compute the lengths of the new segments if we skip a visit.
				lengths_of_new_segments = [compute_length_of_relaxed_dubins_path(individual.states[1].angle_complement(), self.problem_instance.source, self.problem_instance.rho)]

				for idx in range(1, len(individual) - 1):	
					lengths_of_new_segments.append(dubins.shortest_path(individual.states[idx - 1].to_tuple(), individual.states[idx + 1].to_tuple(), self.problem_instance.rho).path_length()) 

				lengths_of_new_segments.append(compute_length_of_relaxed_dubins_path(individual.states[-2], self.problem_instance.sink, self.problem_instance.rho))

				# Compute distances saved & SDR scores for skipping each visit.
				distances_saved = [(individual.segment_lengths[i] + individual.segment_lengths[i + 1]) - lengths_of_new_segments[i] for i in range(len(individual))]
				sdr_scores = [score / distance_saved for score, distance_saved in zip(individual.scores, distances_saved)]

			else:
				# Pop the corresponding elements from the lists.
				lengths_of_new_segments.pop(idx_skipped)
				distances_saved.pop(idx_skipped)
				sdr_scores.pop(idx_skipped)

				m = len(individual)
				if idx_skipped == 0:
					indices_which_needs_updating = [0]
				elif idx_skipped == m:
					indices_which_needs_updating = [m - 1]
				else:
					indices_which_needs_updating = [idx_skipped - 1, idx_skipped]
				
			    # Actually update the values stored in the lists.	
				for idx in indices_which_needs_updating:
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
					sdr_scores[idx] = individual.scores[idx] / distances_saved[idx]

			# Find the index of the visit with the lowest SDR score, which is the index we want to skip.
			# TODO: Investigate if it is better to do this stochastically, i.e. randomly select an index to skip.
			candidates = [i for i in range(len(individual)) if i not in indices_to_skip]
			if len(candidates) == 0: # In this case disregard the fact that the indices should be skipped.
				candidates =  list(range(len(individual)))
				indices_to_skip = []

			idx_skipped = candidates[np.argmin([sdr_score for i, sdr_score in enumerate(sdr_scores) if i in candidates])]

			individual.route.pop(idx_skipped)
			individual.states.pop(idx_skipped)
			individual.scores.pop(idx_skipped)
			individual.aoa_indices.pop(idx_skipped)

			# NOTE: we need to change both the segment length at the idx_skipped and idx_skipped + 1, since we are skipping the 
			#       visit at idx_skipped and this visit is partially responsible for these path segements
			individual.segment_lengths[idx_skipped + 1] = lengths_of_new_segments[idx_skipped]
			individual.segment_lengths.pop(idx_skipped)

			total_distance -= distances_saved[idx_skipped]

		return individual
 
	# ----------------------------------------------------- Parent selection -----------------------------------------------------
	def sigma_scaling(self, c: float = 2) -> Vector:
		"""Implements the sigma scaling algorithm, used to compute the cumulative distribution function (cdf) for parent selection."""
		mean = np.mean(self.fitnesses)
		std_dev = np.std(self.fitnesses)

		if std_dev == 0:
			return np.ones(len(self.population)) / len(self.population)

		scaled_fitnesses = np.clip((self.fitnesses - mean) / std_dev, 0, None)
		return scaled_fitnesses / np.sum(scaled_fitnesses)

	def stochastic_universal_sampling(self, cdf: Vector, m: int) -> List[int]:
		"""Implements the SUS algorithm, used to sample indicies of m parents within the population from the cumulative distribution function (cdf)."""
		indices = [None for _ in range(m)]
		i, j = 0, 0
		r = np.random.uniform(0, 1 / m)

		while i < m:
			while r <= cdf[j]:
				indices[i] = j
				r += 1 / m
				i += 1

			j += 1	
		
		return indices

	# ------------------------------------------------------ Survivor selection -------------------------------------------------------
	def mu_comma_lambda_selection(self, offspring: List[Individual]) -> List[Individual]:
		"""Selects the best mu individuals from the population."""
		if len(offspring) <= self.mu:
			return offspring

		# Sort the population by fitness and return the best mu individuals.
		fitnesses_of_offspring = np.array([self.fitness_function(individual) for individual in offspring])
		indices = np.argsort(self.fitnesses)[:self.mu]
		self.fitnesses = fitnesses_of_offspring[indices]
		return [offspring[i] for i in indices]

	# ------------------------------------------------------ Main algorithm -------------------------------------------------------
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
				while len(offspring) < self.lmbda: 
					parent_indices = self.stochastic_universal_sampling(cdf, m = 2)
					parents = [self.population[i] for i in parent_indices]
					for child in self.modified_partially_mapped_crossover(parents):
						mutated_child, indicies_to_skip = self.mutate(child, p = 0.4, q = 0.3, relative_sigma = 0.9) 
						fixed_mutated_child = self.fix_length(mutated_child, indicies_to_skip)
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
    ga = MemeticAlgorithm(problem_instance, mu = mu, lmbda = mu * 7)
    import cProfile
    cProfile.run("ga.run(300, display_progress_bar = True)", sort = "cumtime")
    route = ga.run(60, display_progress_bar=True, trace=True)
    augmented_route = add_free_visits(problem_instance, route, utility_function)
    print(f"Score of augmented route: {problem_instance.compute_score_of_route(augmented_route, utility_function)} from {problem_instance.compute_score_of_route(route, utility_function)}")
    problem_instance.plot_with_route(route, utility_function)
    plt.show()