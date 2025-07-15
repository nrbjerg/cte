# %%
from __future__ import annotations
from random import shuffle
from copy import deepcopy
from collections import Counter
import matplotlib.lines as mlines
import numpy as np
from typing import List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from math import prod
import matplotlib.pyplot as plt
from classes.data_types import Position, Vector, Matrix
import os

@dataclass()
class Node:
    """Models a target in a orienteering problem."""
    node_id: int
    pos: Position
    score: float
    size: Optional[float] = field(init = False)
    distance_to_sink: Optional[float] = field(init = False)

    def __hash__(self) -> int:
        """Makes the nodes hashable"""
        return self.node_id

    def __eq__ (self, other_node: Node) -> bool:
        return self.node_id == other_node.node_id

    def __repr__ (self) -> str:
        """Returns a string representation of the node"""
        return f"{self.node_id}: ({self.pos[0]:.1f}, {self.pos[1]:.1f}), {self.score:.1f}"

@dataclass
class OPInstance:
    """Stores the data related to a TOP instance."""
    problem_id: str
    t_max: float
    source: Position
    sink: Position
    risks: Matrix
    nodes: List[Node]

    @staticmethod
    def load_from_file(file_name: str) -> OPInstance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "CPM_RTOP", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines())) # NOTE: skip the first line which contains information about the number of nodes.
            N = int(lines[0].split(" ")[-1])
            t_max = float(lines[2].split(" ")[-1]) 

            risk_matrix = np.zeros(shape=(N, N))
            nodes: List[Node] = []
            for node_id, (x_pos, y_pos, score, *risks) in enumerate(map(lambda line: tuple(map(float, line.split(" "))), lines[6:])):
                risk_matrix[node_id] = risks
                pos = np.array([x_pos, y_pos])
                if node_id == 0:
                    source = pos 
                elif node_id == N - 1:
                    sink = pos
                else:
                    nodes.append(Node(node_id - 1, pos, score))
            
            for node in nodes:
                node.distance_to_sink = np.linalg.norm(node.pos - sink) 

            # Compute normalized scores for plotting ect if needed.
            min_score = min([node.score for node in nodes if node.score != 0])
            max_score = max([node.score for node in nodes])

            # Normalized using min-max feature scaling
            node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes] 
            for size, node in zip(node_sizes, nodes):
                node.size = size

            #colors = ["tab:blue", "tab:green", "tab:red", "tab:brown", "tab:pink"] # TODO: add extra colors
            return OPInstance(file_name[:-4], t_max, source, sink, risk_matrix, nodes)

    def compute_score(self, route: List[Node]) -> float:
        """Computes the score of the given route"""
        indicies = [0] + [node.node_id + 1 for node in route] + [len(self.nodes) + 1]
        probability_of_survival = prod([(1 - self.risks[i, j]) for i, j in zip(indicies[:-1], indicies[1:])])
        return probability_of_survival * sum(n.score for n in set(route))

    def compute_length(self, route: List[Node]) -> float:
        """Computes the length of the route """
        if len(route) == 0:
            return np.linalg.norm(self.source - self.sink)

        return (np.linalg.norm(route[0].pos - self.source) + 
                sum([np.linalg.norm(n_i.pos - n_f.pos) for n_i, n_f in zip(route[:-1], route[1:])]) +
                np.linalg.norm(route[-1].pos - self.sink))

    def plot_routes(self, routes: List[Route], alpha: float = 0.1, show: bool = False):
        """Plots a route within the OP"""
        sizes = [node.size for node in self.nodes]
        xs = [node.pos[0] for node in self.nodes]
        ys = [node.pos[1] for node in self.nodes]
        plt.scatter(xs, ys, sizes, color = "black", zorder = 2)

        plt.scatter(*self.source, 50, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 50, marker = "D", c = "black", zorder=4)
        plt.title(f"Instance: {self.problem_id} ($t_{{\\max}} = {self.t_max:.1f}$)")

        for route in routes:
            positions = [self.source] + [node.pos for node in route] + [self.sink]
            for p, q in zip(positions[:-1], positions[1:]):
                plt.plot([p[0], q[0]], [p[1], q[1]], color = "black", alpha = alpha, zorder=0)

        if len(routes) == 1:
            label = f"$S = {self.compute_score(route):.1f}, D = {self.compute_length(route):.1f}$"
            indicators = [mlines.Line2D([], [], color="black", label=label)]
            plt.legend(handles=indicators, loc=1)

        if show:
            plt.show()

# -------------------------------------------------- Solvers ----------------------------------------------- #
Route = List[Node]

import multiprocessing
import random
import time
from typing import List, Tuple, Set

import numpy as np

def greedy(problem: OPInstance, p: float = 1.0) -> Route:
    """Runs a greedy algorithm on the problem instance, with initial routes given"""
    route = []
    remaining_distance = problem.t_max 

    unvisited_nodes = set(problem.nodes)
    
    # Keep going until we reach the sink, in each of the routes.
    while True:
        tail_pos = problem.source if route == [] else route[-1].pos
        last_idx = 0 if route == [] else route[-1].node_id + 1

        # Only look at the nodes which allows us to subsequently go to the sink.
        eligible_nodes = [node for node in unvisited_nodes if 
                          (np.linalg.norm(tail_pos - node.pos) + node.distance_to_sink) <= remaining_distance]
    
        if len(eligible_nodes) == 0:
            return route
        
        else:
            sdr_scores = {node: (node.score * (1 - problem.risks[last_idx][node.node_id + 1]) / np.linalg.norm(tail_pos - node.pos)) for node in eligible_nodes}

            # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
            c = max(sdr_scores.values())
            rcl = [node for node, score in sdr_scores.items() if score >= p * c] 
            node_to_append: Node = np.random.choice(rcl)
            route.append(node_to_append)
            remaining_distance -= np.linalg.norm(tail_pos - node_to_append.pos)

            unvisited_nodes.remove(node_to_append) 
            
def _add_node(problem: OPInstance, route: Route, blacklist: Set[Node]) -> Tuple[Route, float, float]:
    """Finds the best non-visited nodes which can be added to the route, while keeping the distance sufficiently low."""
    remaining_distance = problem.t_max - problem.compute_length(route)
    best_candidate, best_idx_to_insert_candidate = None, None
    best_sdr, best_change_in_distance = 0, None 
    
    # Find the best candidate which can be added to the route
    candidates = set(problem.nodes).difference(blacklist)
    
    positions = [problem.source] + [node.pos for node in route] + [problem.sink]
    for idx, (p, q) in enumerate(zip(positions[:-1], positions[1:])):
        distance_between_p_and_q = np.linalg.norm(p - q)

        # Find the best candidate to add between node0 and node1
        for candidate in candidates:
            distance_through_candidate = np.linalg.norm(p - candidate.pos) + np.linalg.norm(candidate.pos - q)
            change_in_distance = distance_through_candidate - distance_between_p_and_q 
            if change_in_distance == 0:
               best_candidate = candidate
               best_idx_to_insert_candidate, best_change_in_distance = idx, change_in_distance
               break # We have found a node which we can freely add

            if (change_in_distance < remaining_distance) and (candidate.score / change_in_distance > best_sdr):
                best_candidate = candidate
                best_sdr = candidate.score / change_in_distance
                best_idx_to_insert_candidate, best_change_in_distance = idx, change_in_distance

        # If the inner for looop terminates normally, we simply continue otherwise we break out of the current for loop as well 
        else: 
            continue

        break
    
    # Add the candidate to the route
    if best_candidate:
        new_route = route[:best_idx_to_insert_candidate] + [best_candidate] + route[best_idx_to_insert_candidate:]
        return (best_candidate, new_route, best_change_in_distance, best_candidate.score)
    
    else:
        return (None, route, 0, 0) 

def _remove_node(problem: OPInstance, route: Route) -> Tuple[Node | None, Route, float, float]:
    """Tries to remove the worst performing node (measured via the SDR) from the route"""
    worst_srdr = np.inf
    idx_of_worst_candidate = None

    positions = [problem.source] + [node.pos for node in route] + [problem.sink]
    indicies = [0] + [node.node_id for node in route] + [len(problem.nodes) + 1]
    for idx, ((i, j, k), (p, node, q)) in enumerate(zip(zip(indicies[:-2], indicies[1:-1], indicies[2:]), zip(positions[:-2], route, positions[2:]))):
        # Check if node1 should be skiped.
        change_in_distance = np.linalg.norm(p - q) - np.linalg.norm(p - node.pos) - np.linalg.norm(node.pos - q)
        change_in_score = -node.score
        change_in_risk = problem.risks[i][k] - (problem.risks[i][j] + problem.risks[j][k])
        if change_in_distance == 0 and change_in_risk == 0:
            continue
        else:
            srdr = -change_in_score * change_in_risk / change_in_distance
            if srdr < worst_srdr:
                worst_change_in_distance = change_in_distance
                worst_change_in_risk = change_in_risk
                worst_change_in_score = change_in_score
                idx_of_worst_candidate = idx
                worst_sdr = srdr

    if idx_of_worst_candidate:
        new_route = route[:idx_of_worst_candidate] + route[idx_of_worst_candidate + 1:]
        return (route[idx_of_worst_candidate], new_route, worst_change_in_distance, worst_change_in_score)
    else:
        return (None, route, 0, 0)
        
def _hill_climbing(problem: OPInstance, route: Route) -> List[Route]: # TODO: Maybe we could benefit from 2-opt
    """Performs hillclimbing, in order to improve the routes, in the initial solution."""
    # We will never add the same nodes to two distinct routes.
    blacklist = set(route)
    best_route = route
    best_score = problem.compute_score(route)
    best_distance = problem.compute_length(route)

    # Remove nodes while we increase the SDR metric of the route.
    while True:
        removed_node, route, change_in_distance, change_in_score = _remove_node(problem, best_route)

        if (best_score + change_in_score) / (best_distance + change_in_distance) > best_score / best_distance:
            best_route = route
            best_distance += change_in_distance
            best_score += change_in_score
            blacklist.remove(removed_node)

        else: 
            break

    # Add as many nodes as posible to the route, in order to increase the score.
    while (info := _add_node(problem, best_route, blacklist=blacklist))[1] != best_route:
        best_route = info[1]
        if info[0]:
            blacklist.add(info[0])

        else: 
            break

    return best_route

def _baseline_worker_function(data: Tuple[OPInstance, float, int, int | float, int]) -> Route:
    """The worker function which will be used during the next multi processing step, the data tuple contains a seed and the start time."""
    (problem, p, seed, time_budget, start_time) = data

    np.random.seed(seed) 
    random.seed(seed)  

    best_score = 0
    while (time.time() - start_time) < time_budget: 

        route_before_hill_climbing = greedy(problem, p = p) 
        route_after_hill_climbing = _hill_climbing(problem, route_before_hill_climbing)

        if (score := problem.compute_score(route_after_hill_climbing)) > best_score:
            best_score = score
            best_route = route_after_hill_climbing

    return best_route

def grasp(problem: OPInstance, time_budget: int, p: float, trial: int = 0) -> Route:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()

    baseline_route = greedy(problem)

    # Runs multilpe iteraitons in parallel
    number_of_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(number_of_cores) as pool: # TODO: change to number_of_cores
        arguments = [(deepcopy(problem), p, seed, time_budget, start_time) for seed in range(trial * number_of_cores, (trial + 1) * number_of_cores)]
        candidate_routes = [route for route in pool.map(_baseline_worker_function, arguments)] + [baseline_route]
 
    best_route = max(candidate_routes, key = lambda route: problem.compute_score(route))
    return best_route

def main():
    op = load_ROP_instances()[0] 

    print("Running Grasp")
    route = grasp(op, 30.0, 0.7)
    op.plot_routes([route], alpha = 1.0, show = True)

def bench_mark(p: float, time_budget: float = 30.0, n_trials: int = 10):
    """Quickly benchmarks the algorithms on a subset of the chao instances."""
    problems = load_ROP_instances()
    print(f"Benchmarking should take a total of {(time_budget * len(problems) * n_trials / 60):.1f} minutes")

    accumulated_results = {}
    for problem in problems:
        print(f"Benchmarking on {problem.problem_id}")
        results = [problem.compute_score(grasp(problem, time_budget, p, trial = trial)) for trial in range(n_trials)]
 
        accumulated_results[problem.problem_id] = (np.mean(results), np.std(results))

    pass # TODO TODO TODO 

    plt.style.use("ggplot")
    for offset, results, color in zip([-0.12, 0.12], [baseline_results, clustered_results], ["tab:orange", "tab:green"]):
        ys = [results[problem.problem_id][0] for problem in problems]
        stds = [results[problem.problem_id][1] for problem in problems]

        plt.errorbar(np.arange(len(problems)) + offset, ys, stds, linestyle="None", marker="o", color = color)

    ax = plt.gca()
    ax.set_xlim(-0.5, len(problems) - 0.5)
    ax.set_xticks(np.arange(len(problems)))
    ax.set_xticklabels([f"{problem.problem_id[:2]}.1.{problem.problem_id[5]}"for problem in problems], rotation=90)

    plt.ylabel("Score")
    plt.title("Clustered GRASP against Baseline GRASP")
    indicators = [mlines.Line2D([], [], color=color, label=alg, marker = "s") for alg, color in zip(["baseline", "clustered"], ["tab:orange", "tab:green"])]
    plt.legend(handles=indicators, loc=4)

    plt.show()

def load_ROP_instances() -> List[OPInstance]:
    """Loads the set of TOP instances saved within the resources folder."""
    files = ["p4.2.a.0.txt", "p4.2.g.0.txt", "p4.2.m.0.txt", "p4.2.s.0.txt",
             "p7.2.a.0.txt", "p7.2.g.0.txt", "p7.2.m.0.txt", "p7.2.s.0.txt"]
    return [OPInstance.load_from_file(file) for file in files]

if __name__ == "__main__":
    main()
    #bench_mark(8, 3, 0.7)