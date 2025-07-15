import gurobipy as gp
from gurobipy import GRB
import numpy as np 
from typing import List, Set, Callable
from classes.problem_instances.top_instances import TOPInstance
import matplotlib.pyplot as plt 
import pandas as pd 
import os 
from library.TOP_cluster_decomposition.clustering import * 


# -------------------------------------------------------- EXACT SOLVER -------------------------------------------------------- #
def solve_TOP_with_clusters(top_instance: TOPInstance, upper_bound: float, clusters: List[Set[int]], time_budget: float = 3600, show: bool = False) -> float:
    """Solves the TOP with clusters to optimality using GUROBI."""
    model = gp.Model(top_instance.problem_id)
    
    scores = np.array([node.score for node in top_instance.nodes])
    positions = np.array([node.pos for node in top_instance.nodes])
    
    t = np.array([[np.linalg.norm(p - q) for p in positions] for q in positions])

    # Used for indexing the binary decision variables
    number_of_nodes = len(top_instance.nodes) 
    number_of_clusters = len(clusters)
    K = range(top_instance.number_of_agents)
    N = range(number_of_nodes)

    # Setup decision variables
    y = model.addVars(K, N, vtype=GRB.BINARY, name="y")
    x = model.addVars(K, N, N, vtype=GRB.BINARY, name="x")
    u = model.addVars(K, N, vtype=GRB.INTEGER, name="u", lb=2, ub=number_of_nodes)

    # Setup MILP problem
    model.setObjective(gp.quicksum(scores[i] * y[k, i] for k in K for i in N[1:-1]), GRB.MAXIMIZE)

    # Regular TOP constraints
    model.addConstrs((gp.quicksum(y[k, i] for k in K) <= 1 for i in N[1:-1]), name="maximum_of_one_visit")
    model.addConstrs((gp.quicksum(x[k, 0, i] for i in N[1:]) == 1 for k in K), name="leaves_source")
    model.addConstrs((gp.quicksum(x[k, i, number_of_nodes - 1] for i in N[:-1]) == 1 for k in K), name="arrives_at_sink")
    model.addConstrs((gp.quicksum(x[k, i, j] for i in N[:-1]) == y[k, j] for j in N[1:-1] for k in K), name="connectivity_between_visited_nodes1")
    model.addConstrs((gp.quicksum(x[k, j, i] for i in N[1:]) == y[k, j] for j in N[1:-1] for k in K), name="connectivity_between_visited_nodes2")
    model.addConstrs((gp.quicksum(t[i, j] * x[k, i, j] for i in N[:-1] for j in N[1:]) <= top_instance.t_max for k in K), name="maximal_distance_constraint")
    model.addConstrs((u[k, i] - u[k, j] + 1 <= (number_of_nodes - 1) * (1 - x[k, i, j]) for k in K for i in N[1:] for j in N[1:]), name="subtour_elimination")
    
    # If the route arrives at a cluster C_i, it also departs, assuming that i \not \in \{1, M\}
    model.addConstrs(gp.quicksum(x[k, i, j] for i in clusters[m] for j in N if not (j in clusters[m])) == gp.quicksum(x[k, i, j] for i in N for j in clusters[m] if not (i in clusters[m])) for k in K for m in range(1, number_of_clusters - 1))

    # Arrives at each cluster (except C_1 and C_M) at most once
    model.addConstrs(gp.quicksum(x[k, i, j] for i in clusters[m] for j in N if not (j in clusters[m])) <= 1 for k in K for m in range(1, number_of_clusters - 1))
    
    # Never arrive at 1. cluster
    model.addConstrs((gp.quicksum(x[k, i, j] for i in N for j in clusters[0] if not (i in clusters[0])) == 0 for k in K), name="never_depart_from_Mth_cluster")

    # Never depart from M'th cluster
    model.addConstrs((gp.quicksum(x[k, i, j] for i in clusters[-1] for j in N if not (j in clusters[-1])) == 0 for k in K), name="never_depart_from_Mth_cluster")

    # Only added for improved performance (this way unnececary computation is minimized.)
    model.addConstr(gp.quicksum(scores[i] * y[k, i] for k in K for i in N[1:-1]) <= upper_bound, name="TOP_upper_bound")
    # TODO: Set objective bound using the upper bound from TOP 
    #model.setParam(GRB.param.ObjBound, upper_bound)
    model.setParam("TimeLimit", time_budget)
    model.optimize()

    if show:
        for k in K:
            for i in N:
                for j in N:
                    if x[k, i, j].x == 1.0:
                        xs = [positions[i, 0], positions[j, 0]]
                        ys = [positions[i, 1], positions[j, 1]]
                        plt.plot(xs, ys, color = "black")

        for cluster in clusters:
            xs = [positions[i, 0] for i in cluster]
            ys = [positions[i, 1] for i in cluster]
            plt.scatter(xs, ys)

        plt.show()   

    return model.ObjVal, model.MIPGap, model.Runtime

# --------------------------------------------------- EXPERIMENTAL SETUP ------------------------------------------------- #
def run_experiment (top_instance: TOPInstance, upper_bound: float, clustering_func: Callable[[TOPInstance, int], List[Set[int]]], M: int, runs: int = 1, time_budget: float = 600, show: bool = False):
    """Runs tests on the supplied problems and stores the results in a CSV file."""
    clusters = clustering_func(top_instance, M = M)
    lb, gap, dur = solve_TOP_with_clusters(top_instance, upper_bound, clusters, time_budget = time_budget, show = show)

    # Load csv file containing existing results
    path_to_results = os.path.join(os.getcwd(), "new_CDTOP_results.csv")
    df = pd.read_csv(path_to_results) if os.path.isfile(path_to_results) else pd.DataFrame(columns=["Problem Id", "Z_CDTOP", "Z_TOP", "% \\Delta",  "Linkage", "M", "Optimimum"])

    # Save to CSV file
    df.loc[len(df)] = [top_instance.problem_id, lb, upper_bound, round((upper_bound - lb) / upper_bound * 100, 1), clustering_func.__name__, M, dur < time_budget]
    df.to_csv(path_to_results, index=False)

if __name__ == "__main__":
    #Ms = [8, 16, 32] 

    #top_instances_and_upper_bounds = [(TOPInstance.load_from_file("p4.2.a.txt"), 206),
    #                                  (TOPInstance.load_from_file("p4.2.b.txt"), 341),
    #                                  (TOPInstance.load_from_file("p4.2.c.txt"), 452),
    #                                  (TOPInstance.load_from_file("p4.2.d.txt"), 531),
    #                                  # P4.3 Problem 
    #                                  (TOPInstance.load_from_file("p4.3.b.txt"), 38),
    #                                  (TOPInstance.load_from_file("p4.3.c.txt"), 193),
    #                                  (TOPInstance.load_from_file("p4.3.d.txt"), 335),
    #                                  (TOPInstance.load_from_file("p4.3.e.txt"), 468),
    #                                  (TOPInstance.load_from_file("p4.3.f.txt"), 579),
    #                                  (TOPInstance.load_from_file("p4.3.g.txt"), 653),
    #                                  # P4.4 Problem 
    #                                  (TOPInstance.load_from_file("p4.4.d.txt"), 38),
    #                                  (TOPInstance.load_from_file("p4.4.e.txt"), 183),
    #                                  (TOPInstance.load_from_file("p4.4.f.txt"), 324),
    #                                  (TOPInstance.load_from_file("p4.4.g.txt"), 461),
    #                                  (TOPInstance.load_from_file("p4.4.h.txt"), 571)]
    
    #print(len(top_instances_and_upper_bounds) * 3 * 3)
    ##, TOPInstance.load_from_file("p4.3.a.txt"), TOPInstance.load_from_file("p4.4.a.txt")]
    #clustering_functions = [cluster_hc_single_linkage, cluster_hc_average_linkage, cluster_hc_complete_linkage]#, cluster_spectral]
    #for (M, (top_instance, upper_bound), clustering_function) in product(Ms, top_instances_and_upper_bounds, clustering_functions):
    #    run_experiment(top_instance, upper_bound, clustering_function, M, runs = 1)

    CDTOP_instances_with_gaps = [
        #(8, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_single_linkage),
        #(8, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_average_linkage),
        #(8, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_complete_linkage),
        #(8, (TOPInstance.load_from_file("p4.4.g.txt"), 461), cluster_hc_complete_linkage),
        #(8, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_single_linkage),
        #(8, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_average_linkage),
        #(8, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_complete_linkage),
        #(16, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_single_linkage),
        #(16, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_average_linkage),
        #(16, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_complete_linkage),
        #(16, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_average_linkage),
        #(16, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_complete_linkage),
        #(32, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_single_linkage),
        #(32, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_average_linkage),
        #(32, (TOPInstance.load_from_file("p4.3.g.txt"), 653), cluster_hc_complete_linkage),
        #(32, (TOPInstance.load_from_file("p4.4.g.txt"), 461), cluster_hc_single_linkage),
        #(32, (TOPInstance.load_from_file("p4.4.g.txt"), 461), cluster_hc_average_linkage),
        # Actual needed
        (32, (TOPInstance.load_from_file("p4.4.g.txt"), 461), cluster_hc_complete_linkage),
        (32, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_single_linkage),
        (32, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_average_linkage),
        (32, (TOPInstance.load_from_file("p4.4.h.txt"), 571), cluster_hc_complete_linkage),
    ]

    #run_experiment(top_instance=TOPInstance.load_from_file("p4.2.a.txt", needs_plotting=True), upper_bound=206, clustering_func=cluster_hc_average_linkage, M=32, show=True)
    #for (M, (top_instance, upper_bound), clustering_function) in CDTOP_instances_with_gaps:
    #    run_experiment(top_instance, upper_bound, clustering_function, M, runs = 1, time_budget=3600)
    #    
    #run_experiments(top_instance, cluster_hc_single_linkage, M, runs = 2)