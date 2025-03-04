# %%
import itertools
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute, CEDOPADSNode, Visit, UtilityFunction, utility_fixed_optical
import matplotlib.pyplot as plt
from typing import Optional, Set, List, Dict, Tuple, Iterator
from classes.data_types import State, Position, Dir, compute_difference_between_angles, Angle
from library.core.dubins.dubins_api import call_dubins, sample_dubins_path
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path

import dubins 
import numpy as np 

def get_samples(problem: CEDOPADSInstance, k: int, i: int) -> List[Tuple[float, float]]:
    """Generate the samples for the i'th angle interval of the k'th target."""
    theta, phi = problem.nodes[k].thetas[i], problem.nodes[k].phis[i]

    samples = []
    if phi <= np.pi / 2:
        psis = [theta, (theta + phi / 3) % (2 * np.pi), (theta - phi / 3) % (2 * np.pi)]
    else: #phi <= 2 * np.pi / 3:
        psis = [theta, (theta + phi / 4) % (2 * np.pi), (theta - phi / 4) % (2 * np.pi),
                       (theta + phi / 2) % (2 * np.pi), (theta - phi / 2) % (2 * np.pi)]
    
    for psi in psis:
        samples.extend([(psi, (psi + np.pi) % (2 * np.pi)), 
                        (psi, (psi + np.pi + problem.eta / 4) % (2 * np.pi)), 
                        (psi, (psi + np.pi - problem.eta / 4) % (2 * np.pi)),
                        (psi, (psi + np.pi + problem.eta / 2) % (2 * np.pi)),
                        (psi, (psi + np.pi - problem.eta / 2) % (2 * np.pi))])

    return samples

def get_equdistant_samples(problem: CEDOPADSInstance, k: int) -> Iterator[Tuple[Angle, Angle, float]]:
    """Gets equidistant samples (cordinate wise) from within AOA i of node k"""
    r = sum(problem.sensing_radii) / 2
    for i in range(len(problem.nodes[k].AOAs)):
        psis = [(problem.nodes[k].thetas[i] - problem.nodes[k].phis[i] / 4.5) % (2 * np.pi), (problem.nodes[k].thetas[i] + problem.nodes[k].phis[i] / 4.5) % (2 * np.pi)]
        for psi in psis:
            yield (k, psi, (psi + np.pi - problem.eta / 4.5) % (2 * np.pi), r)
            yield (k, psi, (psi + np.pi + problem.eta / 4.5) % (2 * np.pi), r)

def get_close_enough_node_ids(problem: CEDOPADSInstance, unvisited_nodes: Set[int], q0: State, q1: State) -> Set[int]:
    """generate a list of nodes which lies sufficently close to the path segment from q0 to q1, this function is only 
    used to limit the search space of the actual operator""" 
    dubins_path_typ, dubins_path_segment_lengths = call_dubins(q0, q1, problem.rho)
    
    # Check which nodes are sufficently close to the first center point
    if dubins_path_typ[0] == Dir.L: 
        first_center = q0.pos + problem.rho * np.array([np.cos(q0.angle + np.pi / 2), np.sin(q0.angle + np.pi / 2)]) 
    else:
        first_center = q0.pos + problem.rho * np.array([np.cos(q0.angle - np.pi / 2), np.sin(q0.angle - np.pi / 2)]) 

    close_enough_to_first_segment = {node_idx for node_idx in unvisited_nodes if np.linalg.norm(first_center - problem.nodes[node_idx].pos) <= 2 * problem.sensing_radii[1]}

    # Check if the points are close enough to the second segment if it is either a staright line segment or a turn.
    angle_diff = dubins_path_segment_lengths[0] / problem.rho
    heading_after_first_turn = (q0.angle - angle_diff) % (2 * np.pi) 
    if dubins_path_typ[1] == Dir.S:
        close_enough_to_second_segment = set()

        # Compute the end points of the line points, which will be used to compute the minimum length from the straight line segment.
        start_pos = first_center + problem.rho * np.array([np.cos(q0.angle + np.pi / 2 - angle_diff), np.sin(q0.angle + np.pi / 2 - angle_diff)])
        end_pos = first_center + dubins_path_segment_lengths[1] * np.array([np.cos(heading_after_first_turn), np.sin(heading_after_first_turn)])

        for node_idx in unvisited_nodes:
            pos = problem.nodes[node_idx].pos
            # Compute the length between the line segment between position i and position j and the point at position k
            # link to an explaination of the formula: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            area = np.abs((start_pos[1] - end_pos[1]) * pos[0] - (start_pos[0] - end_pos[0]) * pos[1] + start_pos[0] * end_pos[1] - start_pos[1] * end_pos[0])

            distance = area / dubins_path_segment_lengths[1]  

            if distance <= problem.sensing_radii[1]:
                close_enough_to_second_segment.add(node_idx)
    
    else: 
        if dubins_path_typ[1] == Dir.L:
            pos_after_first_turn = first_center + problem.rho * np.array([np.cos(q0.angle + np.pi / 2 - angle_diff), np.sin(q0.angle + np.pi / 2 - angle_diff)]) 
            second_center = pos_after_first_turn + problem.rho *  np.array([np.cos(heading_after_first_turn + np.pi / 2), np.sin(heading_after_first_turn + np.pi / 2)])
        else:
            pos_after_first_turn = first_center + problem.rho * np.array([np.cos(q0.angle - np.pi / 2 + angle_diff), np.sin(q0.angle - np.pi / 2 + angle_diff)]) 
            second_center = pos_after_first_turn + problem.rho * np.array([np.cos(heading_after_first_turn - np.pi / 2), np.sin(heading_after_first_turn - np.pi / 2)])

        close_enough_to_second_segment = {node_idx for node_idx in unvisited_nodes if np.linalg.norm(second_center - problem.nodes[node_idx].pos) <= 2 * problem.sensing_radii[1]}
        

    # Check which nodes are sufficently close to the final center point
    if dubins_path_typ[2] == Dir.L: 
        final_center = q1.pos + problem.rho * np.array([np.cos(q1.angle + np.pi / 2), np.sin(q1.angle + np.pi / 2)]) 
    else:
        final_center = q1.pos + problem.rho * np.array([np.cos(q1.angle - np.pi / 2), np.sin(q1.angle - np.pi / 2)]) 

    close_enough_to_third_segment = {node_idx for node_idx in unvisited_nodes if np.linalg.norm(final_center - problem.nodes[node_idx].pos) < problem.sensing_radii[1]}
    
    return close_enough_to_first_segment.union(close_enough_to_second_segment).union(close_enough_to_third_segment)

# TODO: For now this only works on the non-relaxed dubins paths, an upgrade could be made to get it to work with the relaxed dubins path segments aswell.
def add_free_visits(problem: CEDOPADSInstance, route: CEDOPADSRoute, utility_function: UtilityFunction, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> CEDOPADSRoute:
    """Checks if any visits can be added for free into the route, at between the given indicies, if none are given simply add as many visists as posible."""
    if (start_idx != None) and (end_idx != None):
        assert start_idx < end_idx
    
    else:
        assert (start_idx == None) and (end_idx == None) # NOTE: Only included to avoid confusion.
        start_idx, end_idx = 0, len(route)

    # We may have multiple visits which can be added, hence we need to make sure that we only pick the best ones.
    unvisited_nodes = {k for k in range(len(problem.nodes))}.difference([k for (k, _, _, _) in route])

    candidates: Dict[int, List[Visit]] = {idx: [] for idx in range(start_idx, end_idx)}
    node_ids_of_visits = set()
    states: List[State] = problem.get_states(route[start_idx:end_idx])
    for idx, (q0, q1) in enumerate(zip(states[:-1], states[1:]), start=start_idx):
        # 1. Prune search space 
        close_enough_node_ids = get_close_enough_node_ids(problem, unvisited_nodes, q0, q1)
        #print(close_enough_node_ids)

        # 2. Sample path and check if they are within the zones of aqusion of one or more nodes.
        #print(len(dubins.path_sample(q0.to_tuple(), q1.to_tuple(), problem.rho, problem.sensing_radii[1] - problem.sensing_radii[0])))
        if len(close_enough_node_ids) == 0:
            continue


        #dubins.path_sample(q0.to_tuple(), q1.to_tuple(), problem.rho, (problem.sensing_radii[1] - problem.sensing_radii[0])[0]:
        for configuration in sample_dubins_path(q0, q1, problem.rho, (problem.sensing_radii[1] - problem.sensing_radii[0])):
            #plt.scatter(*configuration.pos, color = "tab:red")
            # Check if each candidate is within the FOV from the configuration and if the configuration lies within one of the AOAs
            for k in close_enough_node_ids:
                delta = problem.nodes[k].pos - configuration.pos 
                r = np.linalg.norm(delta)
                if problem.sensing_radii[0] <= r and r <= problem.sensing_radii[1]:
                    phi = float((np.arctan2(delta[1], delta[0]) + np.pi) % (2 * np.pi))
                    if problem.nodes[k].get_AOA(phi) is not None and compute_difference_between_angles(configuration.angle, (np.pi + phi) % (2 * np.pi)) <= problem.eta / 2:
                        candidates[idx].append((k, phi, configuration.angle, r))
                        node_ids_of_visits.add(k)

    for node_id in node_ids_of_visits:
        # Only include the visit with the highest utility.
        candidates_with_node_id = sum([[(k, phi, tau, r) for (k, phi, tau, r) in candidates[idx] if k == node_id] for idx in range(start_idx, end_idx)], [])
        strongest_candidate_with_node_id = max(candidates_with_node_id, key = lambda visit: problem.compute_score_of_visit(visit, utility_function))

        # Remove each candidate which is not the strongest candidate
        for idx in range(start_idx, end_idx):
            candidates[idx] = [visit for visit in candidates[idx] if visit[0] != node_id or visit == strongest_candidate_with_node_id]

    # Augment the routes with the remaning visits
    augmented_route = route 
    number_of_visits_added = 0
    for idx, visits in candidates.items():
        #for (k, psi, tau, r), color in zip(visits, ["tab:blue", "tab:orange", "tab:red"]):
        #    plt.scatter(*problem.nodes[route[idx][0]].pos, color="tab:purple", zorder=10)
        #    plt.scatter(*problem.nodes[route[idx + 1][0]].pos, color="tab:purple", zorder=10)
        #    plt.scatter(*problem.nodes[k].pos, color = color, zorder=10)
        #    plt.scatter(*problem.nodes[k].get_state(r, psi, tau).pos, zorder=10, color = color)
        #print(f"{augmented_route=}, {idx=}, {visits=}")
        augmented_route = augmented_route[:idx + 1 + number_of_visits_added] + visits + augmented_route[idx + 1 + number_of_visits_added:]
        number_of_visits_added += len(visits)

    return augmented_route

def search_for_psi_tau_and_r(problem: CEDOPADSInstance, route: CEDOPADSRoute, utility_function: UtilityFunction, idx: int) -> CEDOPADSRoute:
    """Optimizes observation angle psi, the heading tau and sensing radius r of the visit at index idx."""
    k, psi, _, r = route[idx]
    aoa = problem.nodes[k].get_AOA(psi)

    valid_rs = [(problem.sensing_radii[0] + r) / 2, r, (problem.sensing_radii[1] + r) / 2]
    valid_psis = [(psi - compute_difference_between_angles(psi, aoa.a) / 2) % (2 * np.pi), psi, (psi + compute_difference_between_angles(psi, aoa.b) / 2) % (2 * np.pi)]

    candidate_visits = [route[idx]]
    # We need to do this because of the fact that each tau depends on the overservation angle psi.
    for (candidate_psi, candidate_r) in itertools.product(valid_psis, valid_rs):
        complement_of_candidate_psi = (np.pi + candidate_psi) % (2 * np.pi) 
        candidate_taus = [(complement_of_candidate_psi - problem.eta / 2) % (2 * np.pi), complement_of_candidate_psi, (complement_of_candidate_psi + problem.eta / 2) % (2 * np.pi)]
        candidate_visits.extend([(k, candidate_psi, candidate_tau, candidate_r) for candidate_tau in candidate_taus])

    route[idx] = max(candidate_visits, key = lambda visit: problem.compute_sdr(visit, route, idx, utility_function))
    return route 

def _add_visit(route: CEDOPADSRoute, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction) -> Tuple[CEDOPADSRoute, float, float]:
    """Finds the best excluded visits which can be added to the route, while keeping the distance sufficiently low."""
    remaining_distance = problem_instance.t_max - problem_instance.compute_length_of_route(route)

    index_of_visit_after_insertion, best_visit = None, None
    best_sdr, best_change_in_distance, best_change_in_score = 0, None, None

    node_indicies = {k for k in range(len(problem_instance.nodes))}.difference([k for k, _, _, _ in route])

    # 1. Check if the visit should be inserted before the route
    q = problem_instance.get_state(route[0])
    original_distance = compute_length_of_relaxed_dubins_path(q.angle_complement(), problem_instance.source, problem_instance.rho)
    for k in node_indicies:
        for i in range(len(problem_instance.nodes[k].thetas)):
            for (psi, tau) in get_samples(problem_instance, k, i):
                visit = (k, psi, tau, problem_instance.sensing_radii[1])
                q_v = problem_instance.get_state(visit)

                #distance_through_visit = np.linalg.norm(node0.pos - candidate.pos) + np.linalg.norm(candidate.pos - node1.pos)
                distance_through_visit = (compute_length_of_relaxed_dubins_path(q_v.angle_complement(), problem_instance.source, problem_instance.rho) +
                                          dubins.shortest_path(q_v.to_tuple(), q.to_tuple(), problem_instance.rho).path_length())

                change_in_distance = distance_through_visit - original_distance 
                score_of_visit = problem_instance.compute_score_of_visit(visit, utility_function)

                if change_in_distance == 0: # We have found a node which we can freely add.
                    return ([visit] + route, 0, score_of_visit)

                if (change_in_distance < remaining_distance) and (score_of_visit / change_in_distance > best_sdr):
                    best_visit = visit
                    best_sdr = score_of_visit / change_in_distance
                    best_change_in_score = score_of_visit
                    index_of_visit_after_insertion, best_change_in_distance = 0, change_in_distance

    # 2. Check if the visit should be inserted in the middle of the route
    for idx in range(len(route) - 1):
        q_i = problem_instance.get_state(route[idx])
        q_f = problem_instance.get_state(route[idx + 1])
        original_distance = dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), problem_instance.rho).path_length()

        # Find the best candidate visit to add between route[idx:] and route[idx:]
        for k in node_indicies:
            for i in range(len(problem_instance.nodes[k].thetas)):
                for (psi, tau) in get_samples(problem_instance, k, i):
                    visit = (k, psi, tau, problem_instance.sensing_radii[1])
                    q_v = problem_instance.get_state(visit)

                    #distance_through_visit = np.linalg.norm(node0.pos - candidate.pos) + np.linalg.norm(candidate.pos - node1.pos)
                    distance_through_visit = (dubins.shortest_path(q_i.to_tuple(), q_v.to_tuple(), problem_instance.rho).path_length() + 
                                              dubins.shortest_path(q_v.to_tuple(), q_f.to_tuple(), problem_instance.rho).path_length())

                    change_in_distance = distance_through_visit - original_distance 
                    score_of_visit = problem_instance.compute_score_of_visit(visit, utility_function)

                    if change_in_distance == 0: # We have found a node which we can freely add.
                        return (route[:idx] + [visit] + route[idx:], 0, score_of_visit)

                    if (change_in_distance < remaining_distance) and (score_of_visit / change_in_distance > best_sdr):
                        best_visit = visit
                        best_sdr = score_of_visit / change_in_distance
                        best_change_in_score = score_of_visit
                        index_of_visit_after_insertion, best_change_in_distance = idx + 1, change_in_distance

    # 3. Check if the visit should be inserted at the end of the route
    q = problem_instance.get_state(route[-1])
    original_distance = compute_length_of_relaxed_dubins_path(q, problem_instance.sink, problem_instance.rho)
    for k in node_indicies:
        for i in range(len(problem_instance.nodes[k].thetas)):
            for (psi, tau) in get_samples(problem_instance, k, i):
                # TODO:
                visit = (k, psi, tau, problem_instance.sensing_radii[1])
                q_v = problem_instance.get_state(visit)

                #distance_through_visit = np.linalg.norm(node0.pos - candidate.pos) + np.linalg.norm(candidate.pos - node1.pos)
                distance_through_visit = (dubins.shortest_path(q.to_tuple(), q_v.to_tuple(), problem_instance.rho).path_length() + 
                                          compute_length_of_relaxed_dubins_path(q_v, problem_instance.sink, problem_instance.rho))

                change_in_distance = distance_through_visit - original_distance 
                score_of_visit = problem_instance.compute_score_of_visit(visit, utility_function)

                if change_in_distance == 0: # We have found a node which we can freely add.
                    return (route + [visit], 0, score_of_visit)

                if (change_in_distance < remaining_distance) and (score_of_visit / change_in_distance > best_sdr):
                    best_visit = visit
                    best_sdr = score_of_visit / change_in_distance
                    best_change_in_score = score_of_visit
                    index_of_visit_after_insertion, best_change_in_distance = len(route), change_in_distance


    # Insert the visit if any visit is found otherwise simply return the route as is, in which case the hill climbing procedure will terminate.
    if index_of_visit_after_insertion != None:
        if index_of_visit_after_insertion == 0:
            return ([best_visit] + route, best_change_in_distance, best_change_in_score)
        elif index_of_visit_after_insertion == len(route):
            return (route + [best_visit], best_change_in_distance, best_change_in_score)
        else: 
            return (route[:index_of_visit_after_insertion] + [best_visit] + route[index_of_visit_after_insertion:], best_change_in_distance, best_change_in_score)

    else:
        return (route, 0, 0)

def remove_visit(route: CEDOPADSRoute, problem_instance: CEDOPADSInstance, utility_function: UtilityFunction) -> Tuple[CEDOPADSRoute, float, float]:
    """Tries to remove the worst performing node (measured via the SDR) from the route"""
    worst_sdr = np.inf
    idx_of_worst_candidate = None

    if len(route) == 1:
        q = problem_instance.get_state(route[0])
        change_in_distance = (np.linalg.norm(problem_instance.source - problem_instance.sink) - 
                              compute_length_of_relaxed_dubins_path(q.angle_complement(), problem_instance.source, problem_instance.rho) - 
                              compute_length_of_relaxed_dubins_path(q, problem_instance.sink, problem_instance.rho))

        return ([], change_in_distance, -problem_instance.compute_score_of_visit(route[0], utility_function))

    for idx, visit in enumerate(route):
        # Check if visit at index idx should be skiped.
        if idx == 0:
            q0, q1 = problem_instance.get_state(visit), problem_instance.get_state(route[idx + 1])
            change_in_distance = (compute_length_of_relaxed_dubins_path(q1.angle_complement(), problem_instance.source, problem_instance.rho) - 
                                  dubins.shortest_path(q0.to_tuple(), q1.to_tuple(), problem_instance.rho).path_length() - 
                                  compute_length_of_relaxed_dubins_path(q0.angle_complement(), problem_instance.source, problem_instance.rho))

        elif idx == len(route) - 1:
            q0, q1 = problem_instance.get_state(route[idx - 1]), problem_instance.get_state(visit)
            change_in_distance = (compute_length_of_relaxed_dubins_path(q0, problem_instance.sink, problem_instance.rho) - 
                                  dubins.shortest_path(q0.to_tuple(), q1.to_tuple(), problem_instance.rho).path_length() - 
                                  compute_length_of_relaxed_dubins_path(q1, problem_instance.sink, problem_instance.rho))

        else:
            q0, q1, q2 = problem_instance.get_state(route[idx - 1]), problem_instance.get_state(visit), problem_instance.get_state(route[idx + 1])
            change_in_distance = (dubins.shortest_path(q0.to_tuple(), q2.to_tuple(), problem_instance.rho).path_length() - 
                                  dubins.shortest_path(q0.to_tuple(), q1.to_tuple(), problem_instance.rho).path_length() -
                                  dubins.shortest_path(q1.to_tuple(), q2.to_tuple(), problem_instance.rho).path_length())

        change_in_score = -problem_instance.compute_score_of_visit(visit, utility_function)
        
        # Needed to check if we need to skip the visit, since it would not decrease the total length anyway
        if change_in_distance == 0:
            print("Change in distance was 0")
            continue

        # NOTE: if the change in score is small (in absolute terms) and the saved distance is great (again in absolute terms),
        #       then the visit should be removed.
        sdr = change_in_score / change_in_distance 
        if sdr < worst_sdr:
            worst_change_in_distance = change_in_distance
            worst_change_in_score = change_in_score
            idx_of_worst_candidate = idx
            worst_sdr = sdr

    if idx_of_worst_candidate != None:
        new_route = route[:idx_of_worst_candidate] + route[idx_of_worst_candidate + 1:]
        return (new_route, worst_change_in_distance, worst_change_in_score)
    else:
        return (route, 0, 0)
        
def hill_climbing(initial_route: CEDOPADSRoute, problem: CEDOPADSInstance, utility_function: UtilityFunction) -> CEDOPADSRoute: 
    """Performs hillclimbing, in order to improve the routes, in the initial solution."""
    # We will never add the same nodes to two distinct routes.
    best_route = initial_route
    best_score = problem.compute_score_of_route(best_route, utility_function)
    best_distance = problem.compute_length_of_route(best_route)

    # Remove nodes while we increase the SDR metric of the route.
    while True:
        route, change_in_distance, change_in_score = remove_visit(best_route, problem, utility_function)
        if (best_score + change_in_score) / (best_distance + change_in_distance) > best_score / best_distance:
            best_route = route
            best_score += change_in_score
            best_distance += change_in_distance
        else: 
            break
            
    # Add as many nodes as posible to the route, in order to increase the score.
    best_route = greedily_add_visits_while_posible(best_route, problem, utility_function, best_distance)

    return best_route 

def greedily_add_visits_while_posible(route: CEDOPADSRoute, problem: CEDOPADSInstance, utility_function: UtilityFunction, distance: float = None):
    """Tries to greedily add new visits to the route, each time picking the one with the highest local SDR score, 
    that is the change in score divided by the change in distance."""
    if distance == None:
        distance = problem.compute_length_of_route(route)

    while (info := _add_visit(route, problem, utility_function))[2] > 0:
        route = info[0]
        distance += info[1]

    return route

#def pairwise_two_opt(route: CEDOPADSRoute, problem_instance: CEDOPADSInstance) -> CEDOPADSRoute:
#    """Performs a pairwise 2-opt heuristic on the route, to decrease the D_I(route)."""
#    for idx, visit0 in enumerate(route[:-1]):
#        visit1 = route[idx + 1]
#
#    return route

if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.c.b.a.txt", needs_plotting = True)
    #cProfile.run("ga.run(60, 3, ga.sigma_scaling, ga.unidirectional_greedy_crossover, ga.mu_comma_lambda_selection, p_c = 1.0, progress_bar = True)", sort = "cumtime")
    route = [(95, 4.076532973079215, 0.7306382368719095, 1.622214567770218), 
             (13, 3.5996164825015127, 0.3697552023573625, 1.5243895692873004), 
             (6, 4.909791903896811, 1.5821360368480857, 1.813105279613803), 
             (66, 4.107897745473237, 0.7816769702149973, 1.5750466855128644), 
             (9, 5.187342028772198, 2.224485906991138, 1.7384712063566754), 
             (79, 4.74176974640164, 1.9719229946483274, 1.7829464594457787), 
             (16, 4.6972496027160595, 1.4838725331480953, 1.7718625316981254), 
             (54, 1.1355156271747044, 4.216094647248483, 1.9589043251852285), 
             (44, 3.8713709359077115, 0.7012530069333316, 1.7550783350969001), 
             (77, 1.7514312896251603, 5.039543001898875, 1.9323738104812922)]
    import copy
    better_route = copy.deepcopy(route)
    #better_route = add_free_visits(problem_instance, copy.deepcopy(route), utility_function)
    #for _ in range(100):
    #    for idx in range(len(better_route)):
    #        better_route = search_for_psi_tau_and_r(problem_instance, better_route, utility_function, idx)

    print(f"{problem_instance.compute_score_of_route(better_route, utility_function)=}, {problem_instance.compute_length_of_route(better_route)}")
    problem_instance.plot_with_route(add_free_visits(problem_instance, route, utility_function), utility_function, color="tab:green")
    problem_instance.plot_with_route(route, utility_function, color="tab:orange")
    plt.show()
    