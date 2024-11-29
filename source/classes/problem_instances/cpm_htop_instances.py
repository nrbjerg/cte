# %%
from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Tuple, Set
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from classes.node import Node, populate_costs
from classes.cpm_route import CPM_Route
import os
from matplotlib import pyplot as plt
from classes.route import Route
from classes.data_types import Matrix
from library.core.interception.intercepter import InterceptionRoute
from math import prod
import matplotlib.lines as mlines

plt.rcParams['figure.figsize'] = [9, 9]
#plt.rcParams['figure.dpi'] = 300

# NOTE: This function takes an interception route to be able to use both a dubins intercepter and an euclidian intercepter.
# TODO: Make this alot more efficient
def compute_CPM_HTOP_scores(problem_instance: CPM_HTOP_Instance, routes: List[Route], cpm_route: InterceptionRoute) -> List[float]:
    """Computes the scores of each UAV and the CPM of a given solution to a CPM-HTOP problem instance."""

    def sigma(k: int, i: int) -> float:
        """Computes the probability of UAV k remaining operational until the CPM has reached interception point i"""
        if k >= len(routes) or k < 0:
            raise ValueError(f"Invalid k={k} supplied!")
        
        if i < len(cpm_route.route_indicies): 
            # In this case compute the probability that UAV k remains operational until the CPM has reached point i
            time_of_ith_interception = cpm_route.time_until_interception(i)
            j = max(idx for idx, visit_time in enumerate(routes[k].visit_times) if visit_time > time_of_ith_interception)
            return prod(1 - problem_instance.risk_matrix[fst.node_id, snd.node_id] for fst, snd in zip(routes[k].nodes[:j - 1], routes[k].nodes[1:j]))

        else:
            # In this case compute the probability that UAV k remains operational until it reaches the sink
            return prod(1 - problem_instance.risk_matrix[fst.node_id, snd.node_id] for fst, snd in zip(routes[k].nodes[:-1], routes[k].nodes[1:]))
        
    # Compute expected score of CPM 
    expected_score_of_cpm = 0
    nodes_whose_scores_have_already_been_extracted_by_cpm = set()
    for i, k in enumerate(cpm_route.route_indicies):
        time_of_ith_interception = cpm_route.time_until_interception(i)

        j = max(idx for idx, visit_time in enumerate(routes[k].visit_times) if visit_time < time_of_ith_interception)
        if j == len(routes[k].nodes) - 1:
            continue

        new_nodes_which_has_been_visited_by_uav = set(routes[k].nodes[:j + 1]).difference(nodes_whose_scores_have_already_been_extracted_by_cpm)
        expected_score_of_cpm += sigma(k, i) * sum(node.score for node in new_nodes_which_has_been_visited_by_uav)
        nodes_whose_scores_have_already_been_extracted_by_cpm = nodes_whose_scores_have_already_been_extracted_by_cpm.union(new_nodes_which_has_been_visited_by_uav)

    # Compute expected remaining scores of UAVs
    expected_scores_of_uavs = []
    for route in routes:
        # NOTE: we do not need sure that we dont count the scores twice if the UAV collects the scores once the route has been terminated.
        #       since we are using the set "nodes_whose_scores_have_already_been_extracted_by_cpm"!
        total_remaining_score = sum(node.score for node in set(route.nodes).difference(nodes_whose_scores_have_already_been_extracted_by_cpm))
        expected_scores_of_uavs.append(sigma(k, len(cpm_route.route_indicies)) * total_remaining_score)

    return expected_scores_of_uavs + [expected_score_of_cpm]

def compute_CPM_HTOP_score(problem_instance: CPM_HTOP_Instance, routes: List[Route], cpm_route: InterceptionRoute) -> float:
    """Computes the scores of each UAV and the CPM of a given solution to a CPM-HTOP problem instance."""
    return sum(compute_CPM_HTOP_scores(problem_instance, routes, cpm_route))

@dataclass
class CPM_HTOP_Instance:
    """Stores the data related to a DTOP-HTOP instance."""
    problem_id: str
    number_of_agents: int
    t_max: float
    cpm_speed: float
    kappa: float
    d_cpm: float
    source: Node
    sink: Node
    nodes: List[Node]
    risk_matrix: Matrix
    _colors: List[str] = None
    _edges_added_to_source_and_sink: Set[Tuple[int, int]] | None = None

    @staticmethod
    def load_from_file(file_name: str, neighbourhood_level: int = 1, needs_plotting: bool = False) -> CPM_HTOP_Instance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "CPM_HTOP", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))
            # NOTE: skip the first line which contains information about the number of nodes.
            N =  int(lines[0].split(" ")[-1])
            number_of_agents = int(lines[1].split(" ")[-1])
            t_max = float(lines[2].split(" ")[-1])
            cpm_speed = float(lines[3].split(" ")[-1]) # TODO: Update to have the CPM have a specific speed 
            kappa = float(lines[4].split(" ")[-1])
            d_cpm = float(lines[5].split(" ")[-1])

            nodes = []
            risk_matrix = np.zeros(shape=(N, N))
            for node_id, (x_pos, y_pos, score, *risks) in enumerate(map(lambda line: tuple(map(float, line.split(" "))), lines[6:])):
                risk_matrix[node_id] = risks
                pos = np.array([x_pos, y_pos])
                nodes.append(Node(node_id, [], pos, score))
            
            # Perform triangulation, according to the neighbourhood level.
            edges_added_to_source_and_sink = CPM_HTOP_Instance._mark_adjacent_nodes_as_adjacent(nodes, neighbourhood_level = neighbourhood_level)

            # Finally make sure that every node is incident to the source and sinks.
            source = nodes[0]
            sink = nodes[-1]

            # Compute and store the distance to the sink, since this will be used repeatedly.
            for i, node in enumerate(nodes):
                nodes[i].distance_to_sink = np.linalg.norm(node.pos - sink.pos)
                
            if needs_plotting:
                # Compute normalized scores for plotting ect if needed.
                min_score = min([node.score for node in nodes if node.score != 0])
                max_score = max([node.score for node in nodes])

                node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 120 for node in nodes[1:-1]] # Normalized using min-max feature scaling
                for size, node in zip(node_sizes, nodes[1:-1]):
                    node.size = size

                # Set colors and return instance
                colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"] # TODO: add extra colors

                return CPM_HTOP_Instance(file_name[:-4], number_of_agents, t_max, cpm_speed, kappa, d_cpm, source, sink, nodes, risk_matrix, _colors = colors, _edges_added_to_source_and_sink = edges_added_to_source_and_sink)

            else: 
                return CPM_HTOP_Instance(file_name[:-4], number_of_agents, t_max, cpm_speed, kappa, d_cpm, source, sink, nodes, risk_matrix)

    @staticmethod
    def _mark_adjacent_nodes_as_adjacent(nodes: List[Node], neighbourhood_level: int = 1) -> Set[Tuple[int, int]]:
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
        edges_added_to_source_and_sink = set()
        for i, node in enumerate(nodes):
            if (i != 0) and (not (nodes[0] in node.adjacent_nodes)):
                nodes[i].adjacent_nodes.append(nodes[0])
                edges_added_to_source_and_sink.add((i, 0)) # We dont want to display these edges 

            if (i != len(nodes) - 1) and (not (nodes[-1] in node.adjacent_nodes)):
                nodes[i].adjacent_nodes.append(nodes[-1])
                edges_added_to_source_and_sink.add((i, len(nodes) - 1)) # We dont want to display these edges 

        return edges_added_to_source_and_sink


    def plot (self, show: bool = True, plot_nodes: bool = True, edges_to_exclude: Set[Tuple[int, int]] = set()):
        """Displays a plot of the problem instance, along with its delauney triangulation"""
        plt.style.use("bmh")
        plt.gca().set_aspect("equal", adjustable="box")

        edges_plotted = set()
        for node in self.nodes: 
            for adjacent_node in node.adjacent_nodes:
                tup = (node.node_id, adjacent_node.node_id)
                if (tup in edges_plotted) or (tup in self._edges_added_to_source_and_sink) or (tup in edges_to_exclude) or (reversed(tup) in edges_to_exclude):
                    continue

                plt.plot([node.pos[0],adjacent_node.pos[0]], [node.pos[1], adjacent_node.pos[1]], 
                         c= "tab:gray", 
                         linewidth=20 * (0.005 + self.risk_matrix[node.node_id, adjacent_node.node_id]),
                         alpha=0.25 + 3 * self.risk_matrix[node.node_id, adjacent_node.node_id], zorder=1)

                # Don't plot the edge going in the opisite direction
                edges_plotted.add((adjacent_node.node_id, node.node_id)) 

        # Plot nodes
        plt.scatter(*self.source.pos, 120, marker = "s", c = "black", zorder=10)
        plt.scatter(*self.sink.pos, 120, marker = "^", c = "black", zorder=10)

        plt.title(f"CPM-HTOP: {self.problem_id}")
        if plot_nodes:
            sizes = [node.size for node in self.nodes[1:-1]] 
            plt.scatter([node.pos[0] for node in self.nodes[1:-1]], [node.pos[1] for node in self.nodes[1:-1]], sizes, c = "tab:gray", zorder=2)

        if show:
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
        edges_to_exclude = set(sum([[(fst.node_id, snd.node_id) for fst, snd in zip(route.nodes[:-1], route.nodes[1:])] for route in routes], []))

        self.plot(show = False, edges_to_exclude=edges_to_exclude)

        for route, color in zip(routes, self._colors):
            for node, adjacent_node in zip(route.nodes[:-1], route.nodes[1:]):
                plt.plot([node.pos[0],adjacent_node.pos[0]], [node.pos[1], adjacent_node.pos[1]], c= color, linewidth=20 * (0.005 + self.risk_matrix[node.node_id, adjacent_node.node_id]), zorder=1)

            if plot_points:
                # NOTE: this works once the route is connected to the sink.
                xs = [node.pos[0] for node in route.nodes]
                ys = [node.pos[1] for node in route.nodes]
                plt.scatter(xs[1:-1], ys[1:-1], [node.size for node in route.nodes[1:-1]], c = color, zorder=3) 

        if show:
            plt.show()

    def plot_CPM_HTOP_solution(self, routes: List[Route], cpm_route: InterceptionRoute, cpm_route_color: str = "tab:purple", show: bool = True):
        """Plots a CPM-HTOP solution, ie. a set of UAV routes and a CPM route"""
        self.plot_with_routes(routes, plot_points=True, show=False)
        cpm_route.plot(show=False)

        scores = compute_CPM_HTOP_scores(self, routes, cpm_route)

        
        indicators = [mlines.Line2D([], [], color=color, label=f"UAV {idx}: {round(score, 2)}", marker="o") for idx, (score, color) in enumerate(zip(scores[:-1], self._colors))]
        indicators.append(mlines.Line2D([], [], color=cpm_route_color, label=f"CPM: {round(scores[-1], 2)}", marker="D"))
        plt.legend(handles=indicators, loc=1)

        if show:
            plt.show()

    def compute_risk_of_route(self, route: Route) -> float:
        """Computes the total risk of traversing a route, using the risk matrix"""
        probability_of_survival = prod(1 - self.risk_matrix[fst.node_id, snd.node_id] for fst, snd in zip(route.nodes[:-1], route.nodes[1:]))
        return 1 - probability_of_survival

def load_CPM_HTOP_instances(needs_plotting: bool = False, neighbourhood_level: int = 1) -> List[CPM_HTOP_Instance]:
    """Loads the set of TOP instances saved within the resources folder."""
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "CPM_HTOP")
    return [CPM_HTOP_Instance.load_from_file(file_name, needs_plotting = needs_plotting, neighbourhood_level = neighbourhood_level) 
            for file_name in os.listdir(folder_with_top_instances)]

if __name__ == "__main__":

    first_cpm_htop_instance = CPM_HTOP_Instance.load_from_file("p4.2.a.0.txt", needs_plotting=True)
    #first_cpm_htop_instance = load_CPM_HTOP_instances(needs_plotting = True, neighbourhood_level = 1)[0]
    first_cpm_htop_instance.plot(show=False)
    plt.savefig("test.png", bbox_inches="tight")
