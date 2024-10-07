# %%
from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Set
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from classes.node import Node, populate_costs
import os
from matplotlib import pyplot as plt
from classes.route import Route

plt.rcParams['figure.figsize'] = [7, 7]

@dataclass
class TOPInstance:
    """Stores the data related to a TOP instance."""
    problem_id: str
    number_of_agents: int
    t_max: float
    source: Node
    sink: Node
    nodes: List[Node]
    _colors: List[str]

    @staticmethod
    def load_from_file(file_name: str, neighbourhood_level: int = 1, needs_plotting: bool = False) -> TOPInstance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "TOP", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))[1:] # NOTE: skip the first line which contains information about the number of nodes.
            number_of_agents = int(lines[0].split(" ")[-1])
            t_max = float(lines[1].split(" ")[-1])

            nodes = []
            for node_id, (x_pos_str, y_pos_str, score_str) in enumerate(map(lambda line: tuple(line.split("\t")), lines[2:])):
                pos = np.array([float(x_pos_str), float(y_pos_str)])
                nodes.append(Node(node_id, [], pos, float(score_str)))
            
            # Perform triangulation, according to the neighbourhood level.
            TOPInstance._mark_adjacent_nodes_as_adjacent(nodes, neighbourhood_level = neighbourhood_level)

            # Finally make sure that every node is incident to the source and sinks.
            source = nodes[0]
            sink = nodes[-1]

            # Compute and store the distance to the sink, since this will be used repeatedly.
            for i, node in enumerate(nodes):
                nodes[i].distance_to_sink = np.linalg.norm(node.pos - sink.pos)
                
            # Compute normalized scores for plotting ect if needed.
            if needs_plotting:
                min_score = min([node.score for node in nodes if node.score != 0])
                max_score = max([node.score for node in nodes])

                node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes[1:-1]] # Normalized using min-max feature scaling
                for size, node in zip(node_sizes, nodes[1:-1]):
                    node.size = size

            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:cyan", "tab:pink", "tab:purple", "tab:brown"] # TODO: add extra colors
            #colors = ["tab:blue", "tab:green", "tab:red", "tab:brown", "tab:pink"] # TODO: add extra colors
            return TOPInstance(file_name[:-4], number_of_agents, t_max, source, sink, nodes, _colors = colors) 
 
    @staticmethod
    def _mark_adjacent_nodes_as_adjacent(nodes: List[Node], neighbourhood_level: int = 1) -> List[Node]:
        """Marks adjacent nodes in the delaunay triangulation as being adjacent, additionally use the neighbourhood level, to add extra edges."""
        positions = [node.pos for node in nodes]

        # Add the edges found in the delaunay triangulation.
        triangulation = Delaunay(positions)
        for simplex in triangulation.simplices: # NOTE: A simplicity is simply an N-dimensional triangle
            for i in simplex:
                for j in simplex:
                    if i == j:
                        continue
                    nodes[i].adjacent_nodes.append(nodes[j])
                    nodes[j].adjacent_nodes.append(nodes[i])
        
        # Make mark nodes adjacent through a neighbourhood_level number of edges as neighbours.
        if neighbourhood_level > 1:
            for i, node in enumerate(nodes):
                adjacent_nodes = node.adjacent_nodes
                adjacent_nodes_at_level = node.adjacent_nodes
                for _ in range(neighbourhood_level - 1):
                    adjacent_nodes_at_level = sum([adjacent_node.adjacent_nodes for adjacent_node in adjacent_nodes_at_level], [])
                    adjacent_nodes.extend(adjacent_nodes_at_level) 
                
                nodes[i].adjacent_nodes = list(set(adjacent_nodes))

        # Make sure that every node is adjacent to the source and sink nodes
        for i, node in enumerate(nodes):
                if (i != 0) and (not (nodes[0] in node.adjacent_nodes)):
                    nodes[i].adjacent_nodes.append(nodes[0])
                
                if (i != len(nodes) - 1) and (not (nodes[-1] in node.adjacent_nodes)):
                    nodes[i].adjacent_nodes.append(nodes[-1])

        return nodes


    def plot (self, show: bool = True, plot_nodes: bool = True):
        """Displays a plot of the problem instance, along with its delauney triangulation"""
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.gca().set_aspect("equal", adjustable="box")

        for node_id, node in enumerate(self.nodes[2:]):
            for adjacent_node in node.adjacent_nodes:
                if adjacent_node.node_id > node_id + 2: # Make sure to only plot each edge once, and also dont plot edges to the sink / source.
                    continue

                # Plot edges.
                plt.plot([node.pos[0], adjacent_node.pos[0]], [node.pos[1], adjacent_node.pos[1]], c = "tab:gray", linestyle="dashed", linewidth=0.5, alpha=0.2, zorder=1)

        # Plot nodes
        plt.scatter(*self.source.pos, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink.pos, 120, marker = "D", c = "black", zorder=4)

        if plot_nodes:
            sizes = [node.size for node in self.nodes[1:-1]] 
            plt.scatter([node.pos[0] for node in self.nodes[1:-1]], [node.pos[1] for node in self.nodes[1:-1]], sizes, c = "tab:gray", zorder=2)

        if show:
            plt.title(f"TOP: {self.problem_id}")
            plt.show()

    def plot_with_zones(self, zones: List[List[Node]], show: bool = True):
        """Plots the top instance with zones."""
        self.plot(show = False, plot_nodes = False)
        
        for zone, color in zip(zones, self._colors):
            sizes = [node.size for node in zone] # NOTE: Each zone includes the source and sink to make path planing easier
            xs = [node.pos[0] for node in zone] 
            ys = [node.pos[1] for node in zone]
            plt.scatter(xs, ys, sizes, c = color, zorder=3)

        if show:
            plt.show()

    def plot_with_routes(self, routes: List[Route], plot_points: bool = False, show: bool = True):
        """Creates a TOP plot with routes."""
        self.plot(show = False)

        node_set = set(sum((route.nodes for route in routes), []))
        total_score = sum(node.score for node in node_set) # NOTE: total score is computed under the assumption 
        total_distance = sum(route.distance for route in routes)

        for route, color in zip(routes, self._colors):
            xs, ys = [], []
            for node in route.nodes:
                xs.append(node.pos[0])
                ys.append(node.pos[1])

            plt.plot(xs, ys, c = color, zorder=3)

            if plot_points:
                # NOTE: this works once the route is connected to the sink.
                plt.scatter(xs[1:-1], ys[1:-1], [node.size for node in route.nodes[1:-1]], c = color, zorder=3) 

        plt.title(f"TOP: {self.problem_id}, Total Score: {round(total_score, 2)}, Total Distance: {round(total_distance, 2)}")
        if show:
            plt.show()

def load_TOP_instances(needs_plotting: bool = False, neighbourhood_level: int = 1) -> List[TOPInstance]:
    """Loads the set of TOP instances saved within the resources folder."""
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "TOP")
    return [TOPInstance.load_from_file(file_name, needs_plotting = needs_plotting, neighbourhood_level = neighbourhood_level) 
            for file_name in os.listdir(folder_with_top_instances)]

if __name__ == "__main__":
    first_top_instance = load_TOP_instances(needs_plotting = True, neighbourhood_level = 2)[0]
    first_top_instance.plot()
    #first_top_instance.plot_with_zones([first_top_instance.nodes[1:20], first_top_instance.nodes[20:40], first_top_instance.nodes[40:60], first_top_instance.nodes[60:80], first_top_instance.nodes[80:-1]])
    #print(first_top_instance.source != first_top_instance.sink)