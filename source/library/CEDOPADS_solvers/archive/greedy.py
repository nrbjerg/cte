# %%
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, load_CEDOPADS_instances
import numpy as np 
import dubins 
from library.core.relaxed_dubins import compute_length_of_relaxed_dubins_path


def greedy(problem: CEDOPADSInstance, sensing_radius: float, rho: float, t_max: float, eta: float, p: float = 1.0) -> CEDOPADSRoute:
    """Runs a greedy algorithm on the problem instance, by iteratively adding a node from the RCL list to the route"""
    # Initialization
    remaining_distance = t_max 
    unvisited_nodes = set(problem.nodes)
    distances_from_sink = {}
    for node in unvisited_nodes:
        for idx_of_theta, theta in enumerate(node.thetas):
            distances_from_sink[(node.node_id, idx_of_theta)] = compute_length_of_relaxed_dubins_path(node.get_state(sensing_radius, theta, (theta + np.pi) % (np.pi * 2)), problem.sink, rho)

    # Figure out the initial node in the route greedily
    route = []
    while True:
        # Compute eligible sdr scores of eligible states
        sdr_scores_of_eligible_states = {}
        for node in unvisited_nodes:
            for idx_of_theta, theta in enumerate(node.thetas):
                q = problem.nodes[node.node_id].get_state(sensing_radius, theta, (theta + np.pi) % (2 * np.pi))
                
                if len(route) == 0:
                    # NOTE: Remember we are computing the length of the "reversed" dubins path.
                    length_of_dubins_path = compute_length_of_relaxed_dubins_path(q.angle_complement(), problem.source, rho)
                else:
                    tail = problem.nodes[route[-1][0]].get_state(sensing_radius, route[-1][1], route[-1][2]) # The last state in the route
                    length_of_dubins_path = dubins.shortest_path(tail.to_tuple(), q.to_tuple(), rho).path_length()
                    
                if length_of_dubins_path < remaining_distance - distances_from_sink[(node.node_id, idx_of_theta)]:
                    sdr_scores_of_eligible_states[(node.node_id, idx_of_theta)] = node.score / length_of_dubins_path

        # If we have no eligible states, simply return the route, since we cannot perform a random greedy choice of the next state.
        if len(sdr_scores_of_eligible_states) == 0:
            return route 
        
        # Compute the reduces candidate list (RCL) and pick a random state as the next state from the list.
        else:
            if p < 1.0:
                number_of_nodes = int(np.ceil(len(sdr_scores_of_eligible_states) * (1 - p)))
            else:
                number_of_nodes = 1
            
            rcl = sorted(sdr_scores_of_eligible_states.keys(), key = lambda info: sdr_scores_of_eligible_states[info], reverse=True)[:number_of_nodes] 

            i = np.random.choice(len(rcl)) 
            (node_id, idx_of_theta) = rcl[i]
            theta = problem.nodes[node_id].thetas[idx_of_theta]
            route.append((node_id, theta, (theta + np.pi) % (2 * np.pi)))
            remaining_distance -= problem.nodes[node_id].score / sdr_scores_of_eligible_states[(node_id, idx_of_theta)] # Gets the length since sdr is score / distance.
            unvisited_nodes.remove(problem.nodes[node_id]) 

if __name__ == "__main__":
    problem: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
    t_max = 200
    rho = 2
    sensing_radius = 2 
    eta = np.pi / 5
    route = greedy(problem, sensing_radius, rho, t_max, eta, p = 1.0)   
    print(route)
    problem.plot_with_route(route, sensing_radius, rho, eta)

