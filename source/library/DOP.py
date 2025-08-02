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

class GA:
    """A genetic algorithm implemented for solving instances of CEDOPADS, using k-point crossover and dynamic mutation."""
    problem_instance: CEDOPADSInstance
    utility_function: UtilityFunction
    mu: int # Population Number
    lmbda: int # Number of offspring
    number_of_nodes: int
    node_indices: Set[int]
    node_indices_array: Vector

    def __init__ (self, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, mu: int = 256, lmbda: int = 256 * 7, seed: Optional[int] = None):
        """Initializes the genetic algorithm including initializing the population."""
        self.problem_instance = problem_instance
        self.utility_function = utility_function
        self.lmbda = lmbda
        self.mu = mu
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
    def pick_new_visit(self, individual: CEDOPADSRoute) -> Visit:
        """Picks a new place to visit, according to how much time is left to run the algorithm."""
        # For the time being simply pick a random node to visit FIXME: Should be updated, 
        # maybe we pick them with a probability which is propotional to the distance to the individual?
        blacklist = set(k for (k, _, _, _) in individual)

        k = random.choice(list(self.node_indices.difference(blacklist)))
        psi = random.choice(self.problem_instance.nodes[k].AOAs).generate_uniform_angle() # TODO:
        tau = (psi + np.pi + np.random.uniform( -self.problem_instance.eta / 2, self.problem_instance.eta / 2)) % (2 * np.pi)
        r = random.uniform(self.problem_instance.sensing_radii[0], self.problem_instance.sensing_radii[1])

        return (k, psi, tau, r)

    def initialize_population (self) -> List[CEDOPADSRoute]:
        """Initializes a random population of routes, for use in the genetic algorithm."""
        population = []
        for _ in range(self.mu):
            route = []
            while self.problem_instance.is_route_feasable(route):
                route.append(self.pick_new_visit(route))

            population.append(route[:-1])

        return population

    # -------------------------- General functions for mutation operators ects ------------------------------ #
    def pick_new_visit_based_on_sdr_and_index(self, individual: CEDOPADSInstance, idx: int, m: int = 5) -> Visit:
        """Picks a new visit based on the ''neighbours'' of the visit as well as the euclidian distance between
           the line segment between the nodes, and the new node, that is the candidate to be added."""
        if idx == 0:
            i = individual[0][0], 
            j = self.number_of_nodes
            q_idx = self.problem_instance.get_state(individual[idx])
            original_distance = compute_length_of_relaxed_dubins_path(q_idx.angle_complement(), self.problem_instance.source, self.problem_instance.rho)
        elif idx == len(individual) - 1:
            i = individual[-1][0]
            j = self.number_of_nodes + 1
            q_idx = self.problem_instance.get_state(individual[idx])
            original_distance = compute_length_of_relaxed_dubins_path(q_idx, self.problem_instance.sink, self.problem_instance.rho)
        else:
            i = individual[idx][0]
            j = individual[idx + 1][0]
            q_idx = self.problem_instance.get_state(individual[idx])
            original_distance = dubins.shortest_path(q_idx.to_tuple(), self.problem_instance.get_state(individual[idx + 1]).to_tuple(), self.problem_instance.rho).path_length()

        self.sdrs_for_overwrite[:] = self.sdr_tensor[i][j][:]
        self.mask_for_overwrite[:] = self.default_mask[:]

        for (k, _, _, _) in individual:
            self.mask_for_overwrite[k] = False

        weights = self.sdrs_for_overwrite[self.mask_for_overwrite]
        probs = weights / np.sum(weights)
        k = np.random.choice(self.node_indices_arrray[self.mask_for_overwrite], p = probs)

        # perform sampling and pick the best one from a set of m posible.
        visits_and_sdrs = []
        for visit in get_equdistant_samples(problem_instance, k):
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
    def fitness_function(self, individual: CEDOPADSRoute) -> float:
        """Computes the fitness of the route, based on the ratio of time left."""
        return self.problem_instance.compute_score_of_route(individual, self.utility_function) 
     
    # -------------------------------------- Recombination Operators -------------------------------------- #
    def partially_mapped_crossover(self, parents: List[CEDOPADSRoute]) -> List[CEDOPADSRoute]:
        """Performs a partially mapped crossover (PMX) operation on the parents to generate two offspring"""
        m = min(len(parent) for parent in parents)
        if not m >= 2:
            return parents 

        indices = sorted(np.random.choice(m, 2, replace = False))

        offspring = []
        for idx in range(2):
            paternal, maternal = parents[idx], parents[(idx + 1) % 2] 
            child = paternal[:indices[0]] + maternal[indices[0]:indices[1]] + paternal[indices[1]:]

            # Fix any mishaps when the child was created, i.e. make sure that the same node i never visited twice.
            ks = {child[jdx][0] for jdx in range(indices[0], indices[1] + 1)} 

            # NOTE: Normaly the order of the visits is reversed, however since our distance "metric" is in this case non-symetric we will not use this convention.
            visits_which_may_replace = [visit for visit in paternal[indices[0]:indices[1]] if visit[0] not in ks]
            number_of_visits_replaced, number_of_visits_which_can_be_replaced = 0, len(visits_which_may_replace)

            indices_to_remove = []
            for jdx in itertools.chain(range(indices[0]), range(indices[1], len(child))):
                # Replace the visit with one from the other parent is posible otherwise simply remove the visit entirely
                if child[jdx][0] in ks:
                    if number_of_visits_replaced < number_of_visits_which_can_be_replaced:
                        child[jdx] = visits_which_may_replace[number_of_visits_replaced]
                        number_of_visits_replaced += 1
                    else:
                        indices_to_remove.append(jdx)

            # actually remove the indices, we have to do it to not mess with the indices.
            for jdx in reversed(indices_to_remove):
                child.pop(jdx)

            offspring.append(child)

        return offspring
    
    #--------------------------------------------- Mutation Operators -------------------------------------- 

    # TODO: we could look at the pheno type which corresponds to the individdual (i.e. what does the actual route look like.)
    #       when inserting new visits into the routes, this could speed up the convergence rate.
    def mutate(self, individual: CEDOPADSRoute, p_s: float, p_i: float, p_r: float, q: float) -> CEDOPADSRoute: 
        """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route.""" 

        # Replace visists along the route.
        n = max(0, min(np.random.geometric(1 - p_r) - 1, len(individual)))
        for i in np.random.choice(len(individual), n, replace = False):
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
                individual.insert(i, self.pick_new_visit_based_on_sdr_and_index(individual, i, i + 1))  
                i += 2
            else:
                i += 1

        if np.random.uniform(0, 1) < p_s and len(individual) >= 2: 
            # Swap two random visists along the route described by the individual.
            indices = np.random.choice(len(individual), size = 2, replace = False)
            individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]

        # Finally we perform multiple angle mutations
        return self.angle_mutation(individual, q)

    # TODO: Mayor performance gains to be had here by storing the lengths of the path segments.
    def fix_length(self, individual: CEDOPADSRoute) -> Tuple[CEDOPADSRoute, float]: 
        """Removes visits from the individual until it obeys the distance constraint."""
        distance = self.problem_instance.compute_length_of_route(individual)
        
        # TODO: This can be optimized heavily if we keep track of the lengths underway instead of recomputing them each time.
        while distance > self.problem_instance.t_max: 
            individual, change_in_distance, _ = remove_visit(individual, self.problem_instance, self.utility_function)
            distance += change_in_distance

        return individual, distance

    # TODO: At somepoint this should be converted to use corrolated mutations.
    def angle_mutation (self, individual: CEDOPADSRoute, q: float) -> CEDOPADSRoute:
        """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
        scale = 0.05 

        # Core idea here is that in the start we are quite interested in making
        # large jumps in each psi and tau, however in the end we are more interested 
        # in wiggeling them around to perform the last bit of optimization
        n = max(0, min(np.random.geometric(1 - q) - 1, len(individual)))
        for idx in np.random.choice(len(individual), n, replace = False):
            (k, psi, _, r) = individual[idx]
            if np.random.uniform(0, 1) < 0.9: # Stay within the same AOA but wiggle the values sligtly.
                # Compute probabilities of picking an areas of acqusion based on their "arc lengths"
                # with higher arc lengths having having  a higher probability of getting chosen.
                arc_lengths = []
                for aoa in self.problem_instance.nodes[k].AOAs:
                    if aoa.contains_angle(psi):
                        break

                # TODO: Consider using beta distribution to compute these
                if aoa.a < aoa.b:
                    new_psi = truncnorm.rvs((aoa.a - psi) / scale, (aoa.b - psi)  / scale, loc = psi, scale = scale)
                else:
                    new_psi = truncnorm.rvs((aoa.a - psi)  / scale, (aoa.b + 2 * np.pi - psi) / scale, loc = psi, scale = scale) % (2 * np.pi)

                new_tau = (new_psi + np.pi + truncnorm.rvs((-self.problem_instance.eta / 2) / scale, (self.problem_instance.eta / 2) / scale,
                                        loc = 0, scale = scale)) % (2 * np.pi)

                new_r = truncnorm.rvs((self.problem_instance.sensing_radii[0] - r) / scale, (self.problem_instance.sensing_radii[1] - r) / scale, loc = r, scale = scale)
                individual[idx] = (k, new_psi, new_tau, new_r)

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
                individual[idx] = (k, new_psi, new_tau, new_r)

        return individual

    # ---------------------------------------- Parent Selection ----------------------------------------- #
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
                
                offspring = []
                while len(offspring) < self.lmbda:
                    parent_indices = self.stochastic_universal_sampling(cdf, m = 2)
                    parents = [self.population[i] for i in parent_indices]
                    for child in self.partially_mapped_crossover(parents): 
                        mutated_child = self.mutate(deepcopy(child), p_s = 0.2, p_i = 0.3, p_r = 0.2, q = 0.1)
                        fixed_mutated_child, _ = self.fix_length(mutated_child) # NOTE: This also returns the length of the mutated_child, which could be usefull in later stages.
                        offspring.append(fixed_mutated_child)

                        # TODO: implement a local search improvement operator, ie. convert
                        #       the algorithm to a Lamarckian memetatic algortihm

                    # NOTE: we are simply interested in optimizing the existing angles within the individual,
                    # hence we create a new offspring where a few of the angles are different compared to 
                    # the original individual, sort of like a Lamarckian mematic algortihm, however
                    # here the local search is done using a stochastic operator.
                    #offspring.append(self.fix_length(self.wiggle_angles(deepcopy(self.population[i]), q = 0.05)))
                    #index_of_individual_to_be_wiggled = parent_indices[np.random.choice(len(parent_indices))]
                    #offspring.append(self.fix_length(self.wiggle_angles(deepcopy(self.population[index_of_individual_to_be_wiggled]), q = 0.2))[0])

                # Survivor selection, responsible for find the lements which are passed onto the next generation.
                gen += 1
                self.population = self.mu_comma_lambda_selection(offspring)
                probabilities = self.sigma_scaling() 
                cdf = np.add.accumulate(probabilities)

                idx_of_individual_with_highest_fitness = np.argmax(self.fitnesses)
                if self.fitnesses[idx_of_individual_with_highest_fitness] > self.highest_fitness_recorded:
                    self.individual_with_highest_recorded_fitness = self.population[idx_of_individual_with_highest_fitness]
                    self.highest_fitness_recorded = self.fitnesses[idx_of_individual_with_highest_fitness]
                    self.length_of_individual_with_highest_recorded_fitness = self.problem_instance.compute_length_of_route(self.individual_with_highest_recorded_fitness)

                # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
                if display_progress_bar:
                    pbar.n = round(time.time() - start_time, 1)
                    avg_fitness = np.mean(self.fitnesses)
                    avg_distance = sum(self.problem_instance.compute_length_of_route(route) for route in self.population) / self.mu
                    length_of_best_route = self.problem_instance.compute_length_of_route(self.population[idx_of_individual_with_highest_fitness])
                    avg_length = sum(len(route) for route in self.population) / self.mu
                    pbar.set_postfix({
                            "Gen": gen,
                            "Best": f"({self.highest_fitness_recorded:.1f}, {self.length_of_individual_with_highest_recorded_fitness:.1f}, {len(self.individual_with_highest_recorded_fitness)})",
                            "Aver": f"({avg_fitness:.1f}, {avg_distance:.1f}, {avg_length:.1f})",
                            "Var": f"{np.var(self.fitnesses):.1f}"
                    })

        return self.individual_with_highest_recorded_fitness

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.f.d.b.txt", needs_plotting = True)
    problem_instance.t_max = 80 
    problem_instance.reduce(12)
    problem_instance.plot({}, show = True)
    mu = 256
    ga = GA(problem_instance, utility_function, mu = mu, lmbda = mu * 7)
    #import cProfile
    #cProfile.run("ga.run(300, display_progress_bar = True)", sort = "cumtime")
    route = ga.run(300, display_progress_bar=True)
    augmented_route = add_free_visits(problem_instance, route, utility_function)
    print(f"Score of augmented route: {problem_instance.compute_score_of_route(augmented_route, utility_function)}")
    problem_instance.plot_with_route(route, utility_function)
    plt.show()