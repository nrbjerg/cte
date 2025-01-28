import dubins
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, CEDOPADSRoute, CEDOPADSNode, Visit, UtilityFunction

def add_to_route(route: CEDOPADSRoute, problem_instance: CEDOPADSInstance) -> CEDOPADSRoute:
    """Adds ''free'' nodes to the route whenever posible, by checking if sampled points 
       along a dubins path segment is within the AOA of different nodes."""
    
    # np.array(dubins.path_sample(q0, q1, self.rho, 0.1)[0])
    states = problem_instance.get_states(route)
    max_delta = problem_instance.sensing_radii[1] - problem_instance.sensing_radii[0]

    # TODO: implement sampling for relaxed dubins. 
    # TODO: maybe we can limit the search space by discarding elements which are to far away from the straight line between the states?
    configurations = sum([dubins.path_sample(q0, q1, problem_instance.rho, max_delta) for q0, q1 in zip(states[1:], states[:-1])], [])

    for (x, y, tau) in configurations:
        # Check if any AOI k is observable from the configuration (x, y, tau)
        
        pass 
    
    return route