# %%
from __future__ import annotations
import matplotlib.lines as mlines
import numpy as np
from typing import List, Optional, Callable, Union, Set
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from classes.data_types import Position, Vector, Matrix
import os

@dataclass()
class Cluster:
    """Models a cluster."""
    id: int
    elements: List[Union[Node, Cluster]]

    @property
    def score(self) -> float:
        """Computes the score of cluster"""
        nodes = self.get_nodes()
        baseline_score = sum(node.score for node in nodes)
        return baseline_score / np.var([node.pos for node in nodes])
            
    @property
    def centroid(self) -> Position:
        """Returns the centroid when the nodes are weighted according to their score."""
        nodes = self.get_nodes()
        return np.sum([node.score * node.pos for node in nodes], axis = 0) / np.sum([node.score for node in nodes])

    def get_nodes(self) -> List[Node]:
        """Returns a list of all of the nodes in the cluster."""
        nodes = []
        for elem in self.elements:
            if type(elem) == Node:
                nodes.append(elem)
            else:
                nodes.extend(elem.get_nodes())

        return nodes

    def get_subclusters(self, depth: int = 1):
        """Gets the children at a given depth in the hierarchical tree."""
        if depth == 0:
            return [self]
        if depth == 1:
            # Are only here for plotting purposes
            return [Cluster(elem.node_id, [elem]) if type(elem) == Node else elem for elem in self.elements]
        else:
            # Are only here for plotting purposes
            sub_clusters = []
            for elem in self.elements:
                if type(elem) == Node:
                    sub_clusters.append(Cluster(elem.node_id, [elem]))
                else:
                    sub_clusters.extend(elem.get_subclusters(depth - 1))
        
            return sub_clusters
    
    def discard_unvisited_clusters(self, visited_clusters: Set[int], depth: int):
        """Removes unvisited clusters from the tree."""
        if depth < 1:
            raise ValueError("We cannot make any cuts if the depth is less than 1")

        else:
            # We make the cuts here
            indicies_to_pop = []
            for (i, elem) in enumerate(self.elements):
                if type(elem) == Cluster and depth != 1:
                    elem.discard_unvisited_clusters(visited_clusters, depth - 1)
                elif (type(elem) == Node and elem.node_id not in visited_clusters) or (type(elem) == Cluster and elem.id not in visited_clusters):
                    indicies_to_pop.append(i)

            # Actually pop the approriate elements
            for i in reversed(indicies_to_pop):
                self.elements.pop(i)

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
    root: Cluster = field(init = False)

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


    @staticmethod
    def generate_instance(n: int) -> OPInstance:
        """Generates an OP instance with n nodes."""
        nodes = []
        for i in range(n):
            pos = np.random.uniform(0, 1, size = 2)
            score = 1 + np.random.exponential(1) 
            nodes.append(Node(i, pos, score))

        source = np.random.uniform(0, 1, size = 2)
        sink = np.random.uniform(0, 1, size = 2)
        t_max = np.linalg.norm(source - sink) * (1 + np.random.uniform(0, 1))

        for node in nodes:
            node.distance_to_sink = np.linalg.norm(node.pos - sink) 

        # Compute normalized scores for plotting ect if needed.
        min_score = min([node.score for node in nodes if node.score != 0])
        max_score = max([node.score for node in nodes])

        # Normalized using min-max feature scaling
        node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes] 
        for size, node in zip(node_sizes, nodes):
            node.size = size

        return OPInstance(f"RG[{n=}, t_max={t_max:.1f}]", t_max, source, sink, nodes)

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

    def plot_with_clusters(self, clusters: List[Cluster], centroids: bool = False, show: bool = False):
        """PLots the TOP instance with the clusters corresponding to the cluster IDs"""
        # Get the Tab20c colormap
        tab20b = plt.get_cmap("tab20b")
        tab20c = plt.get_cmap("tab20c")

        # Extract the 20 colors
        colors = [tab20b(i) for i in range(20)] + [tab20c(i) for i in range(16)]
        
        n_subclusters = 0
        for i, cluster in enumerate(clusters):
            color = colors[i]
            nodes = cluster.get_nodes()
            sizes = [node.size for node in nodes] 
            xs = [node.pos[0] for node in nodes]
            ys = [node.pos[1] for node in nodes]
            plt.scatter(xs, ys, sizes, color = color, zorder=2)
            
            # Plot centroids, if if the cluster is non-trivial
            if len(nodes) != 1 and centroids:
                plt.scatter(*cluster.centroid, color = "black", marker="H", zorder=3)

            n_subclusters += 1

        plt.scatter(*self.source, 50, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 50, marker = "D", c = "black", zorder=4)
        plt.title(f"Instance = {self.problem_id} ($\\# Clusters = {n_subclusters}$)")

        if show:
            plt.show()

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

    def cluster_nodes (self, affinity: Callable[[Vector, Vector], float], linkage: str = "complete"):
        """Clusters the nodes, using the provided threshhold, using heirarchical clustering."""
        data_for_clustering = [node.to_vec() for node in self.nodes]

        custom_metric = lambda mat: pairwise_distances(mat, metric = affinity)
        clustering = AgglomerativeClustering(n_clusters = 1, metric = custom_metric, linkage=linkage).fit(data_for_clustering)
        
        # Generate actual cluster tree
        clusters = [Cluster(node.node_id, [node]) for node in self.nodes]
        for cluster_id, (left, rigth) in enumerate(clustering.children_, len(self.nodes)):
            clusters.append(Cluster(cluster_id, [clusters[left], clusters[rigth]]))
            
        self.root = clusters[-1]

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

        # Only look at the nodes which allows us to subsequently go to the sink.
        eligible_nodes = [node for node in unvisited_nodes if 
                          (np.linalg.norm(tail_pos - node.pos) + node.distance_to_sink) <= remaining_distance]
    
        if len(eligible_nodes) == 0:
            assert len(route) != 0
            return route
        
        else:
            sdr_scores = {node.node_id: (node.score / np.linalg.norm(tail_pos - node.pos)) for node in eligible_nodes}

            # Pick a random node from the RCL list, if p = 1.0, simply use a normal greedy algorithm
            if p < 1.0:
                number_of_nodes = int(np.ceil(len(eligible_nodes) * (1 - p)))
            else:
                number_of_nodes = 1
            
            rcl = sorted(eligible_nodes, key = lambda node: sdr_scores[node.node_id], reverse=True)[:number_of_nodes] # TODO
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
    worst_sdr = np.inf
    idx_of_worst_candidate = None

    positions = [problem.source] + [node.pos for node in route] + [problem.sink]
    for idx, (p, node, q) in enumerate(zip(positions[:-2], route, positions[2:])):
        # Check if node1 should be skiped.
        change_in_distance = np.linalg.norm(p - q) - np.linalg.norm(p - node.pos) - np.linalg.norm(node.pos - q)
        change_in_score = -node.score
        if change_in_distance == 0:
            continue
        else:
            sdr = change_in_score / change_in_distance
            if sdr < worst_sdr:
                worst_change_in_distance = change_in_distance
                worst_change_in_score = change_in_score
                idx_of_worst_candidate = idx
                worst_sdr = sdr

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

def _worker_function(data: Tuple[OPInstance, float, int, int | float, int]) -> List[Route]:
    """The worker function which will be used during the next multi processing step, the data tuple contains a seed and the start time."""
    (problem, p, seed, time_budget, start_time) = data

    np.random.seed(seed) 
    random.seed(seed)  

    best_score = 0
    i = 1
    while (time.time() - start_time) < time_budget: 

        route_before_hill_climbing = greedy(problem, p = p) 
        route_after_hill_climbing = _hill_climbing(problem, route_before_hill_climbing)

        if (score := problem.compute_score(route_after_hill_climbing)) > best_score:
            best_route = route_after_hill_climbing 
            best_score = score
        i += 1

    return best_route

def grasp(problem: OPInstance, time_budget: int, p: float) -> Route:
    """Runs a greedy adataptive search procedure on the problem instance,
    the time_budget is given in seconds and the value p determines how many
    candidates are included in the RCL candidates list,
    which contains a total of 1 - p percent of the candidates"""    
    start_time = time.time()

    baseline_solution = greedy(problem)

    # Runs multilpe iteraitons in parallel
    number_of_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(number_of_cores) as pool:
        arguments = [(problem, p, seed, time_budget, start_time) 
                     for seed in range(1, number_of_cores + 1)]
        candidate_routes = [route for route in pool.map(_worker_function, arguments)] + [baseline_solution]
 
    best_route = max(candidate_routes, key = lambda route: problem.compute_score(route))
    return best_route

def iterative_hierarchical_clustering (op: OPInstance, affinity: Callable[[Vector, Vector], float], ks: List[int], iterative_time_budget: float = 2.0, total_time_budget: float = 30.0) -> Route:
    """Solves the OP described by the given clusters, note that these clusters may simply be nodes."""
    start_time = time.time()
    op.cluster_nodes(affinity)
    m = len(op.nodes)
    for i, k in enumerate(ks):
        # Construct cluster OP
        clusters = op.root.get_subclusters(k)
        N_prime = []
        for cluster in clusters:
            N_prime.append(Node(cluster.id, cluster.centroid, cluster.score))
            N_prime[-1].distance_to_sink = np.linalg.norm(cluster.centroid - op.sink)

        t_max_prime = (1 / (1 + np.exp(-k / np.log2(m)))) * op.t_max
        
        cluster_OP = OPInstance(f"{op.problem_id} [{k=}]", t_max_prime, op.source, op.sink, N_prime)

        # Solve the new OP.
        cluster_OP.plot_with_clusters(clusters, centroids = True)
        route = grasp(cluster_OP, iterative_time_budget, 0.8)

        # Plot route
        cluster_OP.plot_route(route, show = True)

        # Discard unused clusters.
        visited_clusters = {node.node_id for node in route}
        op.root.discard_unvisited_clusters(visited_clusters, depth = k)
        m_prime = len(op.root.get_nodes())
        print(f"Discarded: {m - m_prime} nodes, by looking at a depth of {k=}")
        m = m_prime

    # construct reduced OP 
    reduced_op = OPInstance(f"{op.problem_id} [reduced]", op.t_max, op.source, op.sink, op.root.get_nodes())
    for node in reduced_op.nodes:
        node.distance_to_sink = np.linalg.norm(node.pos - reduced_op.sink)

    final_time_budget = total_time_budget - (time.time() - start_time)
    print(f"Time budget for final GRASP: {final_time_budget:.1f}")
    route = grasp(reduced_op, final_time_budget, 0.8)
    reduced_op.plot_with_clusters(clusters, centroids = False)
    reduced_op.plot_route(route, show = True)

    return route

def main():
    #op = OPInstance.generate_instance(n = 1000)
    op = OPInstance.load_from_file("p4.2.f.txt")
    affinity = lambda v, w: np.linalg.norm(v[:-1] - w[:-1])
    iterative_hierarchical_clustering(op, affinity, [4, 5])

    print("Running Normal Grasp")
    op.cluster_nodes(affinity)
    route_grasp = grasp(op, 30.0, 0.8)
    op.plot_with_clusters([op.root], centroids = False)
    op.plot_route(route_grasp, show = True)

if __name__ == "__main__":
    main()