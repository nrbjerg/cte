from classes.problem_instances.cedopads_instances import CEDOPADSInstance, load_CEDOPADS_instances, CEDOPADSRoute, CEDOPADSNode
from typing import Optional, Set
from classes.data_types import State, Position, Dir
from library.core.dubins.dubins_api import call_dubins
import numpy as np 

def get_close_enough_candidates(problem: CEDOPADSInstance, unvisited_nodes: Set[int], q0: State, q1: State, pos: Position) -> Set[int]:
    """Checks if the point located at position pos lies within a distance of r_max of the dubins path from q0 to q1."""
    dubins_path_typ, dubins_path_segment_lengths = call_dubins(q0, q1, problem.rho)
    
    # Check which nodes are sufficently close to the first center point
    if dubins_path_typ[0] == Dir.L: 
        first_center = q0.pos + problem.rho * np.array([np.cos(q0.angle + np.pi / 2), np.sin(q0.angle + np.pi / 2)]) 
    else:
        first_center = q0.pos + problem.rho * np.array([np.cos(q0.angle - np.pi / 2), np.sin(q0.angle - np.pi / 2)]) 

    close_enough_to_first_segment = {node_idx for node_idx in unvisited_nodes if np.linalg.norm(first_center - problem.nodes[node_idx].pos) < problem.sensing_radii[1]}

    # Check if the points are close enough to the second segment if it is either a staright line segment or a turn.
    angle_diff = dubins_path_segment_lengths[0] / problem.rho
    heading_after_first_turn = (q0.angle - angle_diff) % (2 * np.pi) 
    if dubins_path_typ[1] == Dir.S:
        close_enough_to_second_segment = set()

        # Compute the end points of the line points, which will be used to compute the minimum length from the straight line segment.
        start_pos = first_center + problem.rho * np.array([np.cos(q0.angle + np.pi / 2 - angle_diff), np.sin(q0.angle + np.pi / 2 - angle_diff)])
        end_pos = first_center + dubins_path_segment_lengths[1] * np.array([np.cos(heading_after_first_turn), np.sin(heading_after_first_turn)])

        for node_idx in unvisited_nodes:
            # Compute the length between the line segment between position i and position j and the point at position k
            # link to an explaination of the formula: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            area = np.abs((start_pos[1] - end_pos[1]) * pos[0] - (start_pos[0] - end_pos[0]) * pos[1] + start_pos[0] * end_pos[1] - start_pos[1] * end_pos[0])

            distance = area / dubins_path_segment_lengths[1]  

            if distance < problem.sensing_radii[1]:
                close_enough_to_second_segment.add(node_idx)
    
    else: 
        if dubins_path_typ[1] == Dir.L:
            pos_after_first_turn = first_center + problem.rho * np.array([np.cos(q0.angle + np.pi / 2 - angle_diff), np.sin(q0.angle + np.pi / 2 - angle_diff)]) 
            second_center = pos_after_first_turn + problem.rho *  np.array([np.cos(heading_after_first_turn + np.pi / 2), np.sin(heading_after_first_turn + np.pi / 2)])
        else:
            pos_after_first_turn = first_center + problem.rho * np.array([np.cos(q0.angle - np.pi / 2 + angle_diff), np.sin(q0.angle - np.pi / 2 + angle_diff)]) 
            second_center = pos_after_first_turn + problem.rho * np.array([np.cos(heading_after_first_turn - np.pi / 2), np.sin(heading_after_first_turn - np.pi / 2)])

        close_enough_to_second_segment = {node_idx for node_idx in unvisited_nodes if np.linalg.norm(second_center - problem.nodes[node_idx].pos) < problem.sensing_radii[1]}
        

    # Check which nodes are sufficently close to the final center point
    if dubins_path_typ[2] == Dir.L: 
        final_center = q1.pos + problem.rho * np.array([np.cos(q1.angle + np.pi / 2), np.sin(q1.angle + np.pi / 2)]) 
    else:
        final_center = q1.pos + problem.rho * np.array([np.cos(q1.angle - np.pi / 2), np.sin(q1.angle - np.pi / 2)]) 
    close_enough_to_third_segment = {node_idx for node_idx in unvisited_nodes if np.linalg.norm(first_center - problem.nodes[node_idx].pos) < problem.sensing_radii[1]}
    
    return close_enough_to_first_segment.union(close_enough_to_second_segment).union(close_enough_to_third_segment)

def add_free_visits(problem: CEDOPADSInstance, route: CEDOPADSRoute, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> CEDOPADSRoute:
    """Checks if any visits can be added for free into the route, at between the given indicies, if none are given simply add as many visists as posible."""
    if (start_idx != None) and (end_idx != None):
        assert start_idx < end_idx
    
    else:
        assert (start_idx == None) and (end_idx == None) # NOTE: Only included to avoid confusion.
        start_idx, end_idx = 0, len(route)

    # We may have multiple visits which can be added, hence we need to make sure that we only pick the best ones.
    visits_which_can_be_added = dict()
    unvisited_nodes = {k for k in range(len(problem.nodes))}.difference([k for (k, _, _, _) in route])

    states = problem.get_states()
    for q0, q1 in zip(states[:-1], states[1:]):
        # 1. Prune search space 
        posible_candidates = get_close_enough_candidates(problem, unvisited_nodes, q0, q1)

        # 2. Sample path and check if they are within the zones of aqusion of one or more nodes.

        pass 

    return 
    

