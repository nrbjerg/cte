from __future__ import annotations
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from classes.data_types import Position, Matrix, Vector
import time 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

Route = List[int]

@dataclass()
class Node:
    """Models a target in a orienteering problem."""
    pos: Position
    score: float
    size: Optional[float] = field(init = False)

    def __repr__ (self) -> str:
        """Returns a string representation of the node"""
        return f"({self.pos[0]:.1f}, {self.pos[1]:.1f}), {self.score:.1f}"

@dataclass
class OPInstance:
    """Stores the data related to a TOP instance."""
    problem_id: str
    t_max: float
    source: Position
    sink: Position
    nodes: List[Node]
    distance_matrix: Matrix = field(init = False)
    scores: Vector = field(init = False)
    N: int = field(init = False)

    def __post_init__ (self):
        """Do some final setup ect."""
        # compute distance matrix.
        self.N = len(self.nodes) + 2
        positions = [self.source] + [node.pos for node in self.nodes] + [self.sink]

        self.distance_matrix = np.zeros((self.N, self.N), dtype="float64")
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.distance_matrix[i][j] = np.linalg.norm(positions[i] - positions[j])
                self.distance_matrix[j][i] = self.distance_matrix[i][j]

        self.scores = np.zeros(self.N, dtype="float64")
        self.scores[1:-1] = [node.score for node in self.nodes]

    @staticmethod
    def load_from_file(file_name: str) -> OPInstance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "TOP", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))[1:] # NOTE: skip the first line which contains information about the number of nodes.
            t_max = float(lines[1].split(" ")[-1])

            nodes: List[Node] = []
            for node_id, (x_pos_str, y_pos_str, score_str) in enumerate(map(lambda line: tuple(line.split("\t")), lines[3:-1])):
                pos = np.array([float(x_pos_str), float(y_pos_str)])
                nodes.append(Node(pos, float(score_str)))
            
            # Finally make sure that every node is incident to the source and sinks.
            source = np.array(list(map(float, lines[2].split("\t")))[:-1])
            sink = np.array(list(map(float, lines[-1].split("\t")))[:-1])

            # Compute normalized scores for plotting ect if needed.
            min_score = min([node.score for node in nodes if node.score != 0])
            max_score = max([node.score for node in nodes])

            # Normalized using min-max feature scaling
            node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes] 
            for size, node in zip(node_sizes, nodes):
                node.size = size

            #colors = ["tab:blue", "tab:green", "tab:red", "tab:brown", "tab:pink"] # TODO: add extra colors
            return OPInstance(file_name[:-4], t_max, source, sink, nodes)

    def compute_score(self, route: Route) -> float:
        """Computes the score of the given route"""
        return sum(self.scores[k] for k in route)

    def compute_length(self, route: Route) -> float:
        """Computes the length of the route """
        if len(route) == 0:
            return self.distance_matrix[0, -1]

        return sum(self.distance_matrix[i, j] for i, j in zip(route[:-1], route[1:]))

    def plot_route(self, route: Route, show: bool = False):
        """Plots a route within the OP"""
        plt.style.use("bmh")
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot nodes
        plt.scatter(*self.source, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 120, marker = "D", c = "black", zorder=4)

        for i, node in enumerate(self.nodes, 1):
            if i in route:
                continue
            
            plt.scatter(node.pos[0], node.pos[1], node.size, c = "tab:gray", zorder=2)

        positions = [self.source] + [self.nodes[k - 1].pos for k in route[1:-1]] + [self.sink]
        for p, q in zip(positions[:-1], positions[1:]):
            plt.plot([p[0], q[0]], [p[1], q[1]], color = "tab:orange", zorder=3)

        xs = [self.nodes[k - 1].pos[0] for k in route[1:-1]]
        ys = [self.nodes[k - 1].pos[1] for k in route[1:-1]]
        plt.scatter(xs, ys, [self.nodes[k - 1].size for k in route[1:-1]], c = "tab:orange", zorder=4) 

        if show:
            label = f"$S = {self.compute_score(route):.1f}$, $D = {self.compute_length(route):.1f}$"
            indicators = [mlines.Line2D([], [], color="tab:orange", label=label)]
            plt.legend(handles=indicators, loc=1)
            plt.title(f"{self.problem_id}, ($t_{{max}}: {self.t_max})")
            plt.show()

def solver(problem: OPInstance, time_budget: float = 10, p: float = 0.8) -> Route:
    """Solves the problem instance using a DP and MCTS heuristic"""
    start_time = time.time()

    best_score: float = -np.inf
    best_route: Route = None
    number_of_routes = 0
    while time.time() - start_time < time_budget: # TODO: Maybe we can alter the search based on how much time is left.
        # 1. Construct a route.
        unvisited_nodes = set(range(1, problem.N - 1))
        remaining_distance = problem.t_max
    
        # Keep going until we reach the sink, in each of the routes.
        route = [0]
        while True:
            # Only look at the nodes which allows us to subsequently go to the sink.
            eligible_nodes = [k for k in unvisited_nodes if 
                              problem.distance_matrix[route[-1], k] + problem.distance_matrix[k, -1] <= remaining_distance]

            if len(eligible_nodes) == 0:
                break
    
            sdr_scores = {k: (problem.scores[k] / problem.distance_matrix[route[-1], k]) for k in eligible_nodes}

            # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
            if p < 1.0:
                number_of_nodes = int(np.ceil(len(eligible_nodes) * (1 - p)))
            else:
                number_of_nodes = 1

            rcl = sorted(eligible_nodes, key = lambda k: sdr_scores[k], reverse=True)[:number_of_nodes] # TODO
            k = np.random.choice(rcl)
            remaining_distance -= problem.distance_matrix[route[-1], k]
            route.append(k)

            unvisited_nodes.remove(k) 
        
        route.append(problem.N - 1)

        if (score := sum(problem.scores[k] for k in route[1:-1])) > best_score:
            print(f"Found new best {score=}")
            best_score = score
            best_route = route

        number_of_routes += 1
        # Back propagate.

    print(f"{number_of_routes=}")
    return best_route


if __name__ == "__main__":
    problem = OPInstance.load_from_file("p4.4.m.txt")
    print(f"{problem.t_max=}")
    problem.plot_route(solver(problem), show = True)