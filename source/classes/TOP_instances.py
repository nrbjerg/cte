from __future__ import annotations
from dataclasses import dataclass 
from typing import List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from node import Node, compute_distance
import os
from matplotlib import pyplot as plt
from route import Route

@dataclass
class TOPInstance:
    """Stores the data related to a TOP instance."""
    problem_id: str
    number_of_agents: int
    t_max: float
    source: Node
    sink: Node
    nodes: List[Node]

    @staticmethod
    def load_from_file(file_name: str, needs_plotting: bool = False) -> TOPInstance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "TOP", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))[1:] # NOTE: skip the first line which contains information about the number of nodes.
            number_of_agents = int(lines[0].split(" ")[-1])
            t_max = float(lines[1].split(" ")[-1])

            nodes = []
            for node_id, (x_pos_str, y_pos_str, score_str) in enumerate(map(lambda line: tuple(line.split("\t")), lines[2:])):
                nodes.append(Node(node_id, [], [], (float(x_pos_str), float(y_pos_str)), float(score_str)))
            
            # Perform triangulation
            triangulation = Delaunay([node.pos for node in nodes])
            for simplex in triangulation.simplices: # NOTE: A simplicity is simply an N-dimensional triangle
                for i in simplex:
                    for j in simplex:
                        if i == j:
                            continue
                        nodes[i].adjacent_nodes.append(nodes[j])
                        nodes[j].adjacent_nodes.append(nodes[i])

                        distance = compute_distance(nodes[i], nodes[j])
                        nodes[i].costs.append(distance)
                        nodes[j].costs.append(distance)

            # Finally make sure that every node is incident to the source and sinks
            source = nodes[0]
            sink = nodes[1]
            #for idx, node in enumerate(nodes[2:]):
            #    if source not in node.adjacent_nodes:
            #        nodes[idx].adjacent_nodes.append(source)
            #        nodes[idx].costs.append(compute_distance(node, source))

            #    if sink not in node.adjacent_nodes:
            #        nodes[idx].adjacent_nodes.append(sink)
            #        nodes[idx].costs.append(compute_distance(node, sink))

            # Compute normalized scores for plotting ect if needed.
            if needs_plotting:
                min_score = min([node.score for node in nodes if node.score != 0])
                max_score = max([node.score for node in nodes])

                node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes[2:]] # Normalized using min-max feature scaling
                for size, node in zip(node_sizes, nodes[2:]):
                    node.size = size

            return TOPInstance(file_name[:-4], number_of_agents, t_max, source, sink, nodes) 
 
    def plot (self, show: bool = True):
        """Displays a plot of the problem instance, along with its delauney triangulation"""
        plt.style.use("bmh")

        for node_id, node in enumerate(self.nodes[2:]):
            for adjacent_node in node.adjacent_nodes:
                if adjacent_node.node_id > node_id + 2: # Make sure to only plot each edge once, and also dont plot edges to the sink / source.
                    continue

                # Plot edges.
                plt.plot([node.pos[0], adjacent_node.pos[0]], [node.pos[1], adjacent_node.pos[1]], c = "gray", linestyle="dashed", linewidth=0.5, alpha=0.2, zorder=1)

        # Plot nodes
        plt.scatter(*self.source.pos, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink.pos, 120, marker = "D", c = "black", zorder=4)

        sizes = [node.size for node in self.nodes[2:]] 
        plt.scatter([node.pos[0] for node in self.nodes[2:]], [node.pos[1] for node in self.nodes[2:]], sizes, c = "gray", zorder=2)

        if show:
            plt.title(f"TOP: {self.problem_id}")
            plt.show()

    def plot_with_routes(self, routes: List[Route]):
        """Creates a TOP plot with routes."""
        self.plot(show = False)

        node_set = set(sum((route.nodes for route in routes), []))
        total_score = sum(node.score for node in node_set) # NOTE: total score is computed under the assumption 
        total_cost = sum(route.cost for route in routes)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"] # TODO: add extra colors

        for route, color in zip(routes, colors):
            xs, ys = [], []
            for node in route.nodes:
                xs.append(node.pos[0])
                ys.append(node.pos[1])

            plt.plot(xs, ys, c = color, zorder=3)
            # NOTE: this works once the route is connected to the sink.
            plt.scatter(xs[1:-1], ys[1:-1], [node.size for node in route.nodes[1:-1]], c = color, zorder=3) 

        plt.title(f"TOP: {self.problem_id}, Total Score: {total_score}, Total Cost: {total_cost}")
        plt.show()

def load_TOP_instances(needs_plotting: bool = False) -> List[TOPInstance]:
    """Loads the set of TOP instances saved within the resources folder."""
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "TOP")
    return [TOPInstance.load_from_file(file_name, needs_plotting = needs_plotting) for file_name in os.listdir(folder_with_top_instances)]

if __name__ == "__main__":
    first_top_instance = load_TOP_instances(needs_plotting = True)[0]
    route = Route([first_top_instance.source, first_top_instance.source.adjacent_nodes[0]], [])
    first_top_instance.plot_with_routes([route])

