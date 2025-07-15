import gurobipy as gp
from gurobipy import GRB
import numpy as np 
from typing import List, Set, Callable
from classes.problem_instances.top_instances import TOPInstance
import matplotlib.lines as mlines
import matplotlib.pyplot as plt 
import pandas as pd 
from library.TOP_cluster_decomposition.clustering import * 

def compare_TOP_and_CDTOP(top_instance: TOPInstance, clustering_func: Callable[[TOPInstance, int], List[Set[int]]], M: int, time_budget: float = 200):
    """Generate a plot which shows the difference between the TOP and CDTOP solutions"""
    # -------------------------------------------------------------- SOLVE TOP ----------------------------------------------------------- #
    scores = np.array([node.score for node in top_instance.nodes])
    positions = np.array([node.pos for node in top_instance.nodes])

    t = np.array([[np.linalg.norm(p - q) for p in positions] for q in positions])
    top_model = gp.Model(top_instance.problem_id)
    
    # Used for indexing the binary decision variables
    clusters = clustering_func(top_instance, M)
    number_of_nodes = len(top_instance.nodes) 
    K = range(top_instance.number_of_agents)
    N = range(number_of_nodes)

    # Setup decision variables
    top_y = top_model.addVars(K, N, vtype=GRB.BINARY, name="y")
    top_x = top_model.addVars(K, N, N, vtype=GRB.BINARY, name="x")
    top_u = top_model.addVars(K, N, vtype=GRB.INTEGER, name="u", lb=2, ub=number_of_nodes)

    # Setup MILP problem
    top_model.setObjective(gp.quicksum(scores[i] * top_y[k, i] for k in K for i in N[1:-1]), GRB.MAXIMIZE)

    # Regular TOP constraints
    top_model.addConstrs((gp.quicksum(top_y[k, i] for k in K) <= 1 for i in N[1:-1]), name="maximum_of_one_visit")
    top_model.addConstrs((gp.quicksum(top_x[k, 0, i] for i in N[1:]) == 1 for k in K), name="leaves_source")
    top_model.addConstrs((gp.quicksum(top_x[k, i, number_of_nodes - 1] for i in N[:-1]) == 1 for k in K), name="arrives_at_sink")
    top_model.addConstrs((gp.quicksum(top_x[k, i, j] for i in N[:-1]) == top_y[k, j] for j in N[1:-1] for k in K), name="connectivity_between_visited_nodes1")
    top_model.addConstrs((gp.quicksum(top_x[k, j, i] for i in N[1:]) == top_y[k, j] for j in N[1:-1] for k in K), name="connectivity_between_visited_nodes2")
    top_model.addConstrs((gp.quicksum(t[i, j] * top_x[k, i, j] for i in N[:-1] for j in N[1:]) <= top_instance.t_max for k in K), name="maximal_distance_constraint")
    top_model.addConstrs((top_u[k, i] - top_u[k, j] + 1 <= (number_of_nodes - 1) * (1 - top_x[k, i, j]) for k in K for i in N[1:] for j in N[1:]), name="subtour_elimination")
    
    top_model.setParam("TimeLimit", time_budget / 2)
    top_model.optimize()

    top_score = top_model.ObjVal 

    # -------------------------------------------------------------- SOLVE CDTOP ----------------------------------------------------------- #
    model = gp.Model(top_instance.problem_id)
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
    model.addConstrs(gp.quicksum(x[k, i, j] for i in clusters[m] for j in N if not (j in clusters[m])) == gp.quicksum(x[k, i, j] for i in N for j in clusters[m] if not (i in clusters[m])) for k in K for m in range(1, M - 1))

    # Arrives at each cluster (except C_1 and C_M) at most once
    model.addConstrs(gp.quicksum(x[k, i, j] for i in clusters[m] for j in N if not (j in clusters[m])) <= 1 for k in K for m in range(1, M - 1))
    
    # Never arrive at 1. cluster
    model.addConstrs((gp.quicksum(x[k, i, j] for i in N for j in clusters[0] if not (i in clusters[0])) == 0 for k in K), name="never_depart_from_Mth_cluster")

    # Never depart from M'th cluster
    model.addConstrs((gp.quicksum(x[k, i, j] for i in clusters[-1] for j in N if not (j in clusters[-1])) == 0 for k in K), name="never_depart_from_Mth_cluster")

    # Only added for improved performance (this way unnececary computation is minimized.)
    if top_model.ObjBound == 0.0:
        model.addConstr(gp.quicksum(scores[i] * y[k, i] for k in K for i in N[1:-1]) <= top_score, name="TOP_upper_bound")

    model.setParam("TimeLimit", time_budget / 2)
    model.optimize()

    # Plot TOP routes
    plt.style.use("seaborn-v0_8-ticks")
    for i in N:
        for j in N:
            if i > j:
                continue
            top_use = np.sum(top_x[k, i, j].x + top_x[k, j, i].x for k in K) == 1.0
            cdtop_use = np.sum(x[k, i, j].x + x[k, j, i].x for k in K) == 1.0

            if top_use and cdtop_use:
                print("shared")
                xs = [positions[i, 0], positions[j, 0]]
                ys = [positions[i, 1], positions[j, 1]]
                plt.plot(xs, ys, color = "black", ls = "solid", zorder=0)
            elif top_use: 
                xs = [positions[i, 0], positions[j, 0]]
                ys = [positions[i, 1], positions[j, 1]]
                plt.plot(xs, ys, color = "black", ls = "dotted", zorder=0)
            elif cdtop_use:
                xs = [positions[i, 0], positions[j, 0]]
                ys = [positions[i, 1], positions[j, 1]]
                plt.plot(xs, ys, color = "black", ls = "dashed", zorder=0)

    for cluster in clusters:
        xs = [positions[i, 0] for i in cluster]
        ys = [positions[i, 1] for i in cluster]
        plt.scatter(xs, ys, zorder=2)

    plt.scatter(*positions[0], 80, marker = "s", c = "black", zorder=1)
    plt.scatter(*positions[-1], 80, marker = "D", c = "black", zorder=1)

    plt.title(f"TOP problem id: {top_instance.problem_id}")
    from scipy.spatial import Delaunay

    tri = Delaunay(positions)
    plt.triplot(positions[:, 0], positions[:,1], tri.simplices)

    indicators = [mlines.Line2D([], [], label="TOP", color="black", ls = "dotted"),
                  mlines.Line2D([], [], label="CDTOP", color="black", ls = "dashed"),
                  mlines.Line2D([], [], label="CDTOP & TOP", color="black", ls = "solid")]
    plt.legend(handles=indicators, loc=1)


    plt.show()   


if __name__ == "__main__":
    compare_TOP_and_CDTOP(TOPInstance.load_from_file("p4.2.d.txt"), clustering_func=cluster_hc_average_linkage, M = 16)