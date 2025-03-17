from __future__ import annotations
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from classes.data_types import Position, Vector
import numpy as np 
import os 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

@dataclass()
class Node:
    """Models a target in a orienteering problem."""
    node_id: int
    pos: Position
    score: float
    size: Optional[float] = field(init = False)
    distance_to_sink: Optional[float] = field(init = False)

    def to_vec(self) -> Vector:
        """Returns the nodes as a vector in R^3"""
        return np.array([self.pos[0], self.pos[1], self.score])

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
    nodes: List[Node]

    @staticmethod
    def load_from_file(file_name: str) -> OPInstance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "TOP", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))[1:] # NOTE: skip the first line which contains information about the number of nodes.
            t_max = float(lines[1].split(" ")[-1])

            nodes: List[Node] = []
            for node_id, (x_pos_str, y_pos_str, score_str) in enumerate(map(lambda line: tuple(line.split("\t")), lines[3:-1])):
                pos = np.array([float(x_pos_str), float(y_pos_str)])
                nodes.append(Node(node_id, pos, float(score_str)))
            
            # Finally make sure that every node is incident to the source and sinks.
            source = np.array(list(map(float, lines[2].split("\t")))[:-1])
            sink = np.array(list(map(float, lines[-1].split("\t")))[:-1])

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
            return OPInstance(file_name[:-4], t_max, source, sink, nodes)

    def compute_score(self, route: List[Node]) -> float:
        """Computes the score of the given route"""
        return sum(n.score for n in set(route))

    def compute_length(self, route: List[Node]) -> float:
        """Computes the length of the route """
        if len(route) == 0:
            return np.linalg.norm(self.source - self.sink)

        return (np.linalg.norm(route[0].pos - self.source) + 
                sum([np.linalg.norm(n_i.pos - n_f.pos) for n_i, n_f in zip(route[:-1], route[1:])]) +
                np.linalg.norm(route[-1].pos - self.sink))

    def plot_route(self, route: List[Node], show: bool = False):
        """Plots a route within the OP"""
        positions = [self.source] + [node.pos for node in route] + [self.sink]
        for p, q in zip(positions[:-1], positions[1:]):
            plt.plot([p[0], q[0]], [p[1], q[1]], color = "red", zorder=6)

        if show:
            label = f"$S = {self.compute_score(route):.1f}$, $D = {self.compute_length(route):.1f}$"
            indicators = [mlines.Line2D([], [], color="black", label=label)]
            plt.legend(handles=indicators, loc=1)
            plt.show()

def q_learning_solver(problem: OPInstance, time_budget: float = 30.0, 
                      number_of_episodes: int = 1000, eps: float = 0.1):
    """Tries to solve the orienteering problem using q-learning"""
    # A state will consist of a node id (v: int) & the remaining distance (t_rem: int) we will 
    # only explore feasable routes i.e. routes with non-repeating nodes
    q_table = {}

    for episode in range(number_of_episodes):
        blacklist = set()
        route = []
        for _ in range(len(problem.nodes)):
            pass 

            # Chose an action based on q_table
            q = q_table[route[-1]]

    Q[(v, t), w] =
    pass 
