import gurobipy as gp
from gurobipy import GRB
import numpy as np 
from typing import List, Callable, Tuple, Iterable
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, utility_baseline, utility_fixed_optical, UtilityFunction, Visit, CEDOPADSNode
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from library.CEDOPADS_solvers.samplers import equidistant_sampling
import dubins
from classes.data_types import AreaOfAcquisition, State
import matplotlib.pyplot as plt 

def solve_sampled_CEDOPADS (problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, sampling_method: Callable[[CEDOPADSNode, float, Tuple[float, float]], Iterable[Visit]], time_budget: float = 3600) -> float:
    """Solved a sampled CEDOPADS problem using Gurobi, i.e. where the the headings, observation angles, and observation distances are sampled."""
    model = gp.Model(problem_instance.problem_id)

    # 1. Generate "nodes" of "OP" (i.e. sample the visits around each node.)
    nodes_and_samples = {node.node_id: list(sampling_method(node, problem_instance.eta, problem_instance.sensing_radii)) for node in problem_instance.nodes}
    indices = {}
    start_idx = 0
    for node in problem_instance.nodes:
        indices[node.node_id] = (start_idx, start_idx + len(nodes_and_samples[node.node_id]))
        start_idx += len(nodes_and_samples[node.node_id])

    samples = sum([nodes_and_samples[node.node_id] for node in problem_instance.nodes], [])
    states = problem_instance.get_states(samples)
    scores = [problem_instance.compute_score_of_visit(visit, utility_function) for visit in samples]

    # 2. Setup decision variables
    M = sum(len(samples) for samples in nodes_and_samples.values())
    N = len(problem_instance.nodes)

    y = model.addVars(range(M + 2), vtype=GRB.BINARY, name="y")
    x = model.addVars(range(M + 2), range(M + 2), vtype=GRB.BINARY, name="x")
    u = model.addVars(range(M + 2), vtype=GRB.INTEGER, name="u", lb=2, ub=M + 2)
       
    # 4. Actually compute the distances between samples
    t = np.full((M + 2, M + 2), problem_instance.t_max + 1, dtype="float32")
    for i in range(0, M + 1):
        for j in range(1, M + 2):
            if i == j:
                t[i, j] = 0
            elif (i == 0 and j == M + 1):
                t[i, j] = np.linalg.norm(problem_instance.source - problem_instance.sink)
            elif i == 0:
                t[i, j] = compute_length_of_relaxed_dubins_path(states[j - 1].angle_complement(), problem_instance.source, problem_instance.rho)
            elif j == M + 1:
                t[i, j] = compute_length_of_relaxed_dubins_path(states[i - 1], problem_instance.sink, problem_instance.rho)
            else:
                t[i, j] = dubins.shortest_path(states[i - 1].to_tuple(), states[j - 1].to_tuple(), problem_instance.rho).path_length()
    
    # 5. Add exclusionary constraints, so that no CEDOPADS nodes are visited multiple times.
    model.addConstrs((gp.quicksum(y[i] for i in range(*indices[k])) <= 1 for k in range(1, N - 1)))

    # 6. Setup the problem using an otherwise regular MILP formulation of the OP
    model.setObjective(gp.quicksum(scores[i - 1] * y[i] for i in range(1, M + 1)), GRB.MAXIMIZE)

    # Conforms with the maximal distance constraint.
    model.addConstr(gp.quicksum(t[i, j] * x[i, j] for i in range(M + 1) for j in range(1, M + 2)) <= problem_instance.t_max, name="maximal_distance_constraint")
    
    # Leaves source and arrives at sink
    model.addConstr(gp.quicksum(x[0, i] for i in range(1, M + 2)) == 1, name="leaves_source")
    model.addConstr(gp.quicksum(x[i, M + 1] for i in range(0, M + 1)) == 1, name="arrives_at_sink")

    # Flow conservation
    model.addConstrs((gp.quicksum(x[i, j] for i in range(0, M + 1)) == y[j] for j in range(1, M + 1)), name="connectivity_between_visited_nodes1")
    model.addConstrs((gp.quicksum(x[j, i] for i in range(1, M + 2)) == y[j] for j in range(1, M + 1)), name="connectivity_between_visited_nodes2")
    
    model.addConstrs((u[i] - u[j] + 1 <= (M + 1) * (1 - x[i, j]) for i in range(1, M + 1) for j in range(1, M + 1)), name="subtour_elimination")

    # 7. Actually solve the problem.
    model.setParam("TimeLimit", time_budget)
    model.optimize()

    visit_indices = []
    for i in range(1, M + 1):
        if y[i].x == 1.0:
            visit_indices.append(i)

    route = [visit for _, visit in sorted([(i, samples[i - 1]) for i in visit_indices], key=lambda pair: u[pair[0]].x)]
        
    return route, model.ObjVal, model.MIPGap, model.Runtime

if __name__ == "__main__":
    utility_function = utility_fixed_optical
    problem_instance: CEDOPADSInstance = CEDOPADSInstance.load_from_file("p4.4.g.c.a.txt", needs_plotting = True)
    route, obj_value, MIP_gap, run_time = solve_sampled_CEDOPADS(problem_instance, utility_function = utility_fixed_optical, sampling_method = equidistant_sampling, time_budget = 300)
    print(route, obj_value, MIP_gap, run_time)
    problem_instance.plot_with_route(route, utility_fixed_optical, show=True)
    plt.show()