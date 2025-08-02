import gurobipy as gp
from gurobipy import GRB
import numpy as np 
from typing import List, Callable, Tuple, Iterable
from classes.problem_instances.cedopads_instances import CEDOPADSInstance, utility_baseline, utility_fixed_optical, UtilityFunction, Visit, CEDOPADSNode
from library.core.dubins.relaxed_dubins import compute_length_of_relaxed_dubins_path
from classes.data_types import AreaOfAcquisition, State
import matplotlib.pyplot as plt 

def simple_sampling(node: CEDOPADSNode, sampling_radii: Tuple[float, float]) -> Iterable[Visit]:
    pass 

def solve_sampled_CEDOPADS (problem_instance: CEDOPADSInstance, utility_function: UtilityFunction, sampling_method: Callable[[CEDOPADSNode, Tuple[float, float]], Iterable[Visit]], time_budget: float = 3600) -> float:
    """Solved a sampled CEDOPADS problem using gurobi, i.e. where the the headings, observation angles, and observation distances are sampled."""
    model = gp.Model(problem_instance.problem_id)

    # 1. Generate "nodes" of "OP" (i.e. sample the visits around each node.)
    nodes_and_samples = {node.node_id: list(simple_sampling(node, problem_instance.sensing_radii)) for node in problem_instance.nodes}
    indices = {}
    start_idx = 1
    for node in problem_instance.nodes:
        indices[node.node_id] = (start_idx, start_idx + len(nodes_and_samples[node.node_id]))
        start_idx = len(nodes_and_samples[node.node_id])

    samples = sum([nodes_and_samples[node.node_id] for node in problem_instance.nodes], [])
    states = problem_instance.get_states(samples)
    scores = [0] + [problem_instance.compute_score_of_visit(visit) for visit in samples] + [0]

    # 2. Setup decision variables
    number_of_nodes = sum(len(samples) for samples in nodes_and_samples.values()) + 2 # 0 is the source and number_of_nodes - 1 is the sink.
    N = range(number_of_nodes + 2)

    y = model.addVars(N, vtype=GRB.BINARY, name="y")
    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
    u = model.addVars(N, vtype=GRB.INTEGER, name="u", lb=2, ub=number_of_nodes + 2)
       
    # 4. Actually compute the distances between samples
    t = np.zeros((len(problem_instance.nodes) + 2, len(problem_instance.nodes) + 2), dtype="float32")
    for i in N:
        for j in N:
            if i == j or i == number_of_nodes:
                continue

            if (i == number_of_nodes - 1 and j == 0) or (i == 0 and j == number_of_nodes - 1):
                t[i, j] = np.linalg.norm(problem_instance.source - problem_instance.sink)

            elif i == number_of_nodes - 1:
                t[i, j] = states[j - 1]


    # 5. Add exclusionary constraints, so that no CEDOPADS nodes are visited multiple times.
    model.addConstrs((gp.quicksum(y[i] for i in range(*indices[node.node_id])) <= 1 for node in problem_instance.nodes))

    # 6. Setup the problem using an otherwise regular MILP formulation of the OP



    # 7. Actually solve the problem.
    model.setParam("TimeLimit", time_budget)
    model.optimize()

    return model.ObjVal, model.MIPGap, model.Runtime