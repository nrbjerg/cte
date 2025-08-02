# %%
import dubins
import random
import cProfile
import time
from typing import List, Optional, Callable, Set, Tuple
import numpy as np
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute, Visit
from library.core.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import State, Angle, AngleInterval
from numpy.typing import ArrayLike
import hashlib
from tqdm import tqdm
from itertools import combinations_with_replacement, combinations
import matplotlib.pyplot as plt

def fitness_function(individual: CEDOPADSRoute, problem: CEDOPADSInstance, sensing_radius: float, rho: float, t_max: float, eta: float, ratio_of_time_used: float) -> float:
    """Computes the fitness of the route, based on the ratio of time left."""
    # We wish to add a penalty to penalize routes which are to long
    # and to make the penalty more and more suvere as time is running out.
    length = problem.compute_length_of_route(individual, sensing_radius, rho)
    length_penalty = 1 / (1 + np.exp(-max(ratio_of_time_used * (length - t_max), 0)))

    score = problem.compute_score_of_route(individual, eta)
    return 2 * (1 - length_penalty) * score

def pick_new_visit(individual: CEDOPADSRoute, node_indices: Set[int], problem: CEDOPADSInstance, eta: float) -> Visit:
    """Picks a new place to visit, according to how much time is left to run the algorithm."""
    # For the time being simply pick a random node to visit FIXME: Should be updated, 
    # maybe we pick them with a probability which is propotional to the distance to the individual?
    blacklist = set(k for (k, _, _) in individual)

    k = random.choice(list(node_indices.difference(blacklist)))
    psi = random.choice(problem.nodes[k].intervals).generate_uniform_angle()
    tau = (psi + np.pi + np.random.uniform( - eta / 2, eta / 2)) % (2 * np.pi)

    return (k, psi, tau)

def generate_initial_population(problem: CEDOPADSInstance, t_max: float, rho: float, sensing_radius: float, mu: int, eta: float) -> List[CEDOPADSRoute]:
    """Generates an initial population for the genetic algorithm."""
    # Seed RNG for repeatability
    hash_of_id = hashlib.sha1(problem.problem_id.encode("utf-8")).hexdigest()[:8]
    seed = int(hash_of_id, 16)
        
    np.random.seed(seed)
    random.seed(seed)
        
    # Make sure that the sink can be reached from the source otherwise there is no point in running the algorithm.
    if (d := np.linalg.norm(problem.source - problem.sink)) > t_max:
        raise ValueError(f"Cannot construct a valid dubins route since ||source - sink|| = {d} is greater than t_max = {t_max}")

    # Generate an initial population, which almost satifies the distant constraint, ie. add nodes until
    # the sink cannot be reached within d - t_max units where d denotes the length of the route.
    node_indices = set(range(len(problem.nodes)))
    population = []
    for _ in range(mu):
        route = []
        while len(route) == 0 or problem.compute_length_of_route(route, sensing_radius, rho) < t_max:
            route.append(pick_new_visit(route, node_indices, problem, eta))

        population.append(route[:-1])
    
    return population


# ----------------------------------------- Survivor Selection -------------------------------------- #
#def mu_plus_lambda_selection(self, offspring: List[CEDOPADSRoute], parents: List[CEDOPADSRoute], fitnesses_of_parents: ArrayLike, ratio_of_time_used: float) -> List[CEDOPADSRoute]:
#    """Merges the newly generated offspring with the generation and selects the best mu individuals to survive until the next generation."""
#    # NOTE: Recall that the fitness is time dependent hence we need to recalculate it for the parents.
#    fitnesses_of_parents_and_offspring = np.array(list(map(lambda child: fitness_function(child, ratio_of_time_used), parents + offspring)))
#
#    # Calculate the indices of the best performing memebers
#    indices_of_new_generation = np.argsort(fitnesses_of_parents_and_offspring)[-mu:]
#    
#    # Simply set the new fitnesses which have been calculated recently.
#    fitnesses = fitnesses_of_parents_and_offspring[indices_of_new_generation]
#
#    return [self.population[i] if i < self.mu else offspring[i - self.mu] for i in indices_of_new_generation]

def mu_comma_lambda_selection(offspring: List[CEDOPADSRoute], problem: CEDOPADSInstance, sensing_radius: float, rho: float, t_max: float, eta: float, ratio_of_time_used: float, mu: int) -> Tuple[ArrayLike, List[CEDOPADSRoute]]:
    """Picks the mu offspring with the highest fitnesses to populate the the next generation."""
    fitnesses_of_offspring = np.array(list(map(lambda child: fitness_function(child, problem, sensing_radius, rho, t_max, eta, ratio_of_time_used), offspring)))

    # Calculate the indices of the best performing memebers
    indices_of_new_generation = np.argsort(fitnesses_of_offspring)[-mu:]
    
    return (fitnesses_of_offspring[indices_of_new_generation], [offspring[i] for i in indices_of_new_generation])

#--------------------------------------------- Mutation Operators -------------------------------------- #
def mutate(problem, node_indices: Set[int], individual: CEDOPADSRoute, p_s: float, p_id: float, p_r: float, q0: float, q1: float, eta: float) -> CEDOPADSRoute: 
    # Example arguments for p onwards: 0.3, 0.2, 0.6, 0.02, 0.5
    """Copies and mutates the copy of the individual, based on the time remaining biasing the removal of visits if the route is to long, and otherwise adding / replacing visits within the route."""
    u = np.random.uniform(0, 1)
    if u < p_s and len(individual) > 1: 
        # Swap two random visists along the route described by the individual.
        indices = np.random.choice(len(individual), size = 2, replace = False)
        individual[indices[0]], individual[indices[1]] = individual[indices[1]], individual[indices[0]]

    # Remove random visists along the route, if posible.
    while np.random.uniform(0, 1) < p_id and len(individual) > 0:
        idx = np.random.choice(len(individual), size = 1)[0]
        individual.pop(idx)

    # Replace visists along the route.
    if len(individual) != 0 and np.random.uniform(0, 1) < p_r:
        idx = np.random.choice(len(individual), size = 1)[0]
        individual[idx] = pick_new_visit(individual, node_indices, problem, eta)

    # Insert a random number of new visits into the route.
    while np.random.uniform(0, 1) < p_id:
        new_visit = pick_new_visit(individual, node_indices, problem, eta)

        if len(individual) == 0:
            individual = [new_visit]

        else:
            idx = np.random.choice(len(individual), size = 1)[0]
            individual.insert(idx, new_visit)

    return angle_mutation(problem, individual, q0, q1, eta)

def angle_mutation(problem: CEDOPADSInstance, individual: CEDOPADSRoute, q0: float, q1: float, eta: float) -> CEDOPADSRoute:
    """Mutates the angles of the individual by sampling from a uniform or trunced normal distribution."""
    assert q0 + q1 <= 1

    for idx, (k, psi, _) in enumerate(individual):
        u = random.uniform(0, 1)
        if u < q0:
            # Compute probabilities of picking an angle intervals based on their "arc lengths"
            # with higher arc lengths having having  a higher probability of getting chosen.
            arc_lengths = []
            for interval in problem.nodes[k].intervals:
                arc_length = interval.b - interval.a if interval.a < interval.b else (2 * np.pi - interval.b) + interval.a
                arc_lengths.append(arc_length)

            interval_probabilities = np.array(arc_lengths) / sum(arc_lengths)
            interval: AngleInterval = np.random.choice(problem.nodes[k].intervals, size = 1, p = interval_probabilities)[0]
            
            # Generate angles from interval
            new_psi =  interval.generate_uniform_angle()

        elif u > q0 and u <= q1 + q0: # NOTE: Core idea is that we use this to "wiggle" the weights to slightly increase the fitness
            
            # Figure out which interval the current angle is located within.
            for interval in problem.nodes[k].intervals:
                if interval.contains(psi):
                    break

            # Set bounds for random number generation NOTE: There are checks for these in the data_types.py file.
            new_psi = interval.generate_uniform_angle()
            
        else: 
            # We keep the allele as is!
            continue 

        # Generate a new tau non-uniformly based on the scale_for_tau argument
        # new_tau = truncnorm.rvs(-self.eta / (2 * scale_for_tau), self.eta / (2 * scale_for_tau), 
        #                         loc = np.pi + new_psi, scale = scale_for_tau) % (2 * np.pi)
        new_tau = np.random.uniform(np.pi + new_psi - eta / 2, np.pi + new_psi + eta / 2) % (2 * np.pi)
        individual[idx] = (k, new_psi, new_tau)
        
    return individual

def multi_index_greedy_crossover(problem: CEDOPADSInstance, parents: List[CEDOPADSRoute], node_indices: Set[int], k: int, n: int, eta: float, sensing_radius: float, rho: float) -> List[CEDOPADSRoute]:
    """Performs order crossover, by iteratively selecting two parents and producing their offspring."""
    # Compute offspring for each pair of parents included in the parents list.
    offspring = []
    scores = [[problem.nodes[k].compute_score(psi, tau, eta) for (k, psi, tau) in parent] for parent in parents]
    states = [[problem.nodes[k].get_state(sensing_radius, psi, tau) for (k, psi, tau) in parent] for parent in parents]

    number_of_candidates = k * n
    # NOTE: these repeatedly gets overwritten each time a new greedy choice is made.
    sdr_of_visits = np.zeros(number_of_candidates)
    distances = np.zeros(number_of_candidates)

    for indices in combinations(range(len(parents)), k):
        pool = [parents[index] for index in indices]
        
        for i, dominant_parent in enumerate(pool):
            # Keep track of the current index in each of the parents.
            child = [dominant_parent[0]]
            index = 1
            seen = {dominant_parent[0][0]}

            # Perform a greedy choice of the next visit in the route based on the sdr score, 
            # from the final state in the route to the next candidate state.
            q = problem.nodes[dominant_parent[0][0]].get_state(sensing_radius, dominant_parent[0][1], dominant_parent[0][1])
            total_distance = compute_length_of_relaxed_dubins_path(q.angle_complement() , problem.source, rho)
            while total_distance + compute_length_of_relaxed_dubins_path(q, problem.sink, rho) < t_max:
                candidates = []

                for idx, (index, parent) in enumerate(zip(indices, pool)):
                    # Make sure that every route in the pool has an appropriate length and that we have not already visited the greedily chosen node.
                    for jdx, l in enumerate(range(index, index + n)):
                        if l >= len(parent) or parent[l][0] in seen:
                            new_visit = pick_new_visit(child, node_indices, problem, eta)
                            distances[idx * n + jdx] = problem.compute_distance_between_state_and_visit(q, new_visit, sensing_radius, rho) 
                            sdr_of_visits[idx * n + jdx] = problem.compute_score_of_visit(new_visit, eta) / distances[idx * n + jdx]
                            candidates.append(new_visit)

                        else:
                            distances[idx * n + jdx] = dubins.shortest_path(q.to_tuple(), states[index][l].to_tuple(), rho).path_length()
                            sdr_of_visits[idx * n + jdx] = scores[index][l] / distances[idx * n + jdx]
                            candidates.append(parent[l])

                # Chose the next visit greedily, based on the sdr score.
                probs = sdr_of_visits / np.sum(sdr_of_visits)
                i = np.random.choice(number_of_candidates, p = probs) 
                child.append(candidates[i])
                total_distance += distances[i]
                seen.add(child[-1][0])
                q = problem.nodes[child[-1][0]].get_state(sensing_radius, child[-1][1], child[-1][2])
                index += 1

            offspring.append(child[:-1])

    return offspring

# ---------------------------------------- Parent Selection ----------------------------------------- #
def sigma_scaling(fitnesses: ArrayLike, c: float = 2) -> ArrayLike:
    """Uses sigma scaling to compute the probabilities that each indvidual in the population is chosen for recombination."""
    mean = np.mean(fitnesses)
    std_var = np.std(fitnesses)
    modified_fitnesses = np.maximum(fitnesses - (mean - c * std_var), 0)
    return modified_fitnesses / np.sum(modified_fitnesses)

# --------------------------------------- Sampling of parents --------------------------------------- #
def stochastic_universal_sampling (population: List[CEDOPADSRoute], cdf: ArrayLike, m: int) -> List[CEDOPADSRoute]:
    """Implements the SUS algortihm, used to sample m parents from the population."""
    parents = [None for _ in range(m)] 
    i, j = 0, 0
    r = np.random.uniform(0, 1 / m)

    # The core idea behind the algorithm is that instead of doing m-roletewheels to find m parents, 
    # we do a single spin with a roletewheel with m equally spaced indicators (ie. we pick r once and
    # the offset for each of the indicators is 1 / m)
    while i < m:
        while r <= cdf[j]:
            parents[i] = population[j]
            r += 1 / m
            i += 1
        
        j += 1

    return parents
#    def windowing(fitnesses) -> ArrayLike:
        #"""Uses windowing to compute the probabilities that each indvidual in the population is chosen for recombination."""
        #minimum_fitness = np.min(self.fitnesses)
        #modified_fitnesses = self.fitnesses - minimum_fitness
        #return modified_fitnesses / np.sum(modified_fitnesses)


    #def exponential_ranking(self) -> ArrayLike:
        #"""Uses expoential ranking to compute the probabilities that each indvidual in the population is chosen for recombination."""
        #rankings = np.argsort(self.fitnesses)
        #modified_fitnesses = 1 - np.exp(-rankings)
        #return modified_fitnesses / np.sum(modified_fitnesses)


    #def roulette_wheel (self, cdf: ArrayLike, m: int) -> List[CEDOPADSRoute]:
        #"""Implements roulette wheel sampling, to sample m parents from the population."""
        #parents = []

        ## Basically we just sample the population according to the probabilities given indirectly by the CDF
        ## which corresponds to spinning m roulette wheels where the lengths of each segment is proportional to
        ## the probability of picking each of the individuals of the population.
        #for _ in range(m):
            #r = np.random.uniform(0, 1)
            #i = 0
            #while r <= cdf[i] or i == self.mu - 1:
                #i += 1
            
            #parents.append(self.population[i - 1])
            
        #return parents


#-------------------------------------------- Main Loop of GA --------------------------------------- #
def run(problem: CEDOPADSInstance, time_budget: float, m: int, parent_selection_mechanism: Callable[[], List[CEDOPADSRoute]], crossover_mechanism: Callable[[List[CEDOPADSRoute]], List[CEDOPADSRoute]], survivor_selection_mechanism: Callable[[List[CEDOPADSRoute], float], ArrayLike], mu, sensing_radius, rho, t_max, eta) -> CEDOPADSRoute:
    """Runs the genetic algorithm for a prespecified time, given by the time_budget, using k-point crossover with m parrents."""
    start_time = time.time()
    population = generate_initial_population(problem, t_max, rho, sensing_radius, mu, eta)
    node_indices = set(range(len(problem.nodes)))

    # Compute fitnesses of each route in the population and associated probabilities using 
    # the parent_selection_mechanism passed to the method, which sould be one of the parent
    # selection methods defined earlier in the class declaration.
    fitnesses = [fitness_function(individual, problem, sensing_radius, rho, t_max, eta, 0) for individual in population]

    # NOTE: The fitnesses are set as a field of the class so they are accessable within the method
    probabilities = parent_selection_mechanism(fitnesses) 
    cdf = np.add.accumulate(probabilities)

    i = 0
    best_route, highest_score = None, 0
    with tqdm(total=time_budget, desc="GA: ") as pbar:
        # Set initial progress bar information.
        idx_of_best_route = np.argmax(fitnesses)
        pbar.set_postfix({
                "Gen": i,
                "Best": f"({fitnesses[idx_of_best_route]:.1f}, {problem.compute_length_of_route(population[idx_of_best_route], sensing_radius, rho):.1f})",
                "Aver": f"({np.sum(fitnesses) / mu:.1f}, {sum(problem.compute_length_of_route(route, sensing_radius, rho) for route in population) / mu:.1f})"
            })

        while (elapsed_time := (time.time() - start_time)) < time_budget:
            ratio_of_time_used = elapsed_time / time_budget

            # Generate new offspring using parent selection based on the computed fitnesses, to select m parents
            # and perform k-point crossover using these parents, to create a total of m^k children which is
            # subsequently mutated according to the mutation_probability, this also allows the "jiggeling" of 
            # the angles in order to find new versions of the parents with higher fitness values.
            offspring = []
            for _ in range(mu):
                parents = stochastic_universal_sampling(population, cdf, m = m)
                
                # NOTE: If children equal their parents we need to make sure that we still have sufficient variation in the genetic pool.
                for child in crossover_mechanism(problem, parents, node_indices, 2, 2, eta, sensing_radius, rho): 
                    mutated_child = mutate(problem, node_indices, child, 0.3, 0.5, 0.3, 0.3, 0.5, eta)
                    if len(mutated_child) > 0:
                        offspring.append(mutated_child)
                    else:
                        offspring.append(child)

            # Replace the worst performing individuals based on their fitness values, 
            # however do it softly so that the population increases over time
            # Survivor selection mechanism (Replacement)
            i += 1
            fitnesses, population = survivor_selection_mechanism(offspring, problem, sensing_radius, rho, t_max, eta, ratio_of_time_used, mu)
            probabilities = parent_selection_mechanism(fitnesses) 
            cdf = np.add.accumulate(probabilities)

            # Update the tqdm progress bar and add extra information regarding the genetic algorithm.
            pbar.n = round(time.time() - start_time, 1)

            idx_of_best_route = np.argmax(fitnesses)
            if (problem.compute_length_of_route(population[idx_of_best_route], sensing_radius, rho) < t_max and
                fitnesses[idx_of_best_route] > highest_score):
                best_route = population[idx_of_best_route]
                highest_score = fitnesses[idx_of_best_route]

            pbar.set_postfix({
                    "Gen": i,
                    "Best": f"({fitnesses[idx_of_best_route]:.1f}, {problem.compute_length_of_route(population[idx_of_best_route], sensing_radius, rho):.1f})",
                    "Aver": f"({np.sum(fitnesses) / mu:.1f}, {sum(problem.compute_length_of_route(route, sensing_radius, rho) for route in population) / mu:.1f})"
            })
    
    # TODO: Finally try to fix routes which are to long before returning the best route
    permitable_routes = list(filter(lambda route: problem.compute_length_of_route(route, sensing_radius, rho) < t_max, population))
    best_permitable_route = max(permitable_routes, key = lambda route: problem.compute_score_of_route(route, eta))
    if highest_score < problem.compute_score_of_route(best_permitable_route, eta):
        return best_permitable_route
    else:
        return best_route

if __name__ == "__main__":
    problem: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
    t_max = 200 
    rho = 2
    sensing_radius = 2
    eta = np.pi / 5
    route = run(problem, 60, 3, sigma_scaling, multi_index_greedy_crossover, mu_comma_lambda_selection, 512, sensing_radius, rho, t_max, eta)
    #route = ga.run(60, 4, ga.sigma_secaling, ga.multi_index_greedy_crossover, ga.mu_comma_lambda_selection)
    #print(f"Rounte score: {problem.compute_score_of_route(route, eta):.1f}, Route length: {problem.compute_length_of_route(route, sensing_radius, rho):.1f}")
    #problem.plot_with_route(route, sensing_radius, rho, eta)
    #plt.plot()
    #states = problem.get_states(route, sensing_radius)

