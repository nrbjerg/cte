"""
This file contains description of a heuristic for decomposing TOP problems into disjoint subproblems.

Based on the cluster decomposition TOP. 
"""

import matplotlib.pyplot as plt
from classes.problem_instances.top_instances import TOPInstance
import numpy as np
from typing import List, Set, Tuple, Dict, Union
from dataclasses import dataclass
from classes.data_types import Matrix, Vector
from scipy.spatial import Delaunay
from library.TOP_cluster_decomposition.clustering import * 
from itertools import product

@dataclass
class ConcreteSubProblem: 
    """Models a concrete subproblem, from the CDTOP"""
    positions: Matrix 
    scores: Vector 
    K: int 

    # All of the below should have a length of K with each index corresponding to a specific agent.
    sources: List[List[int]] 
    sinks: List[List[int]]

@dataclass
class GenericSubProblem:
    """Models a generic subproblem from the CDTOP."""
    positions: Matrix 
    scores: Vector 
    source_and_sink_candidates: Dict[int, List[int]] 

    def convert_to_concrete_subproblem(self, prev_cluster_indicies: List[int], next_cluster_indices: List[int], t_maxes: List[float]) -> ConcreteSubProblem:
        """Converts the generic subproblem into a concrete subproblem"""
        K = len(prev_cluster_indicies)
        return ConcreteSubProblem(self.positions, self.scores, K, t_maxes, [self.source_and_sink_candidates[m] for m in prev_cluster_indicies], [self.source_and_sink_candidates[m] for m in next_cluster_indices])

def plot_sub_problem(sub_problem: Union[GenericSubProblem, GenericSubProblem], color: str, marker: str, score_bounds: Tuple[float, float] = None):
    """Plots a generic or concrete subproblem"""
    sizes = [(0.2 + (score - score_bounds[0]) / (score_bounds[1] - score_bounds[0])) * 100 for score in sub_problem.scores] # NOTE: Normalized using min-max feature scaling
    
    if type(sub_problem) == GenericSubProblem:
        candidates = sum(sub_problem.source_and_sink_candidates.values(), [])
        edge_colors = ["black" if i in candidates else color for i in range(len(sizes))]
        markers = ["H" if i in candidates else "o" for i in range(len(sizes))]
        line_widths = [1.8 if i in candidates else 0 for i in range(len(sizes))]
    
    else:
        sources = sum(sub_problem.sources, [])
        sinks = sum(sub_problem.sinks, [])

        edge_colors = ["black" if (i in sources) or (i in sinks) else color for i in range(len(sizes))]
        line_widths = [1.8 if (i in sources) or (i in sinks) else 0 for i in range(len(sizes))]
        markers = []
        for i in range(len(sizes)):
            if (i in sources) and (i in sinks):
                markers.append("H")
            elif i in sources:
                markers.append("s")
            elif i in sinks:
                markers.append("D")
            else:
                markers.append("o")
        
    plt.scatter(sub_problem.positions[:, 0], sub_problem.positions[:, 1], c = color, marker = markers, edgecolors = edge_colors, linewidths=line_widths, s = sizes, zorder=1)

@dataclass
class GenericSubProblemGraph: 
    """Models a graph consisting of generic subproblems"""
    positions: Matrix
    scores: Vector
    inter_cluster_edges: List[Tuple[int, int]]
    generic_sub_problems: List[GenericSubProblem]
    clusters: List[List[int]]

    def __init__ (self, top_instance: TOPInstance, clusters: List[List[int]]):
        """Generates a sub problem graph bsaed on a given TOP instance & a list of clusters."""
        M = len(clusters)
        vertex_to_cluster_map = {i: m for m, cluster in enumerate(clusters) for i in cluster}
        
        self.positions = np.array([node.pos for node in top_instance.nodes])
        self.scores = np.array([node.score for node in top_instance.nodes])
        self.clusters = clusters

        # Compute Delunary triangulation for setting up the cluster graph & find inter cluster edges.
        tri = Delaunay(self.positions)
   
        self.inter_cluster_edges = {m: [] for m in range(M)}
        self.intra_cluster_edges = {m: [] for m in range(M)}

        for simplex in tri.simplices:
            for (v, u) in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])):
                c_v = vertex_to_cluster_map[v]
                c_u = vertex_to_cluster_map[u]
                if c_v == c_u:
                    self.intra_cluster_edges[c_v].append((v, u))
                else:
                    self.inter_cluster_edges[c_v].append((v, u))
                    self.inter_cluster_edges[c_u].append((u, v))

        # Construct subproblems
        self.generic_sub_problems = [] 
        for m0, cluster in enumerate(clusters):
            source_and_sink_candidates = {}
            for (v, u) in self.inter_cluster_edges[m0]:
                source_and_sink_candidates[vertex_to_cluster_map[u]] = source_and_sink_candidates.get(vertex_to_cluster_map[u], []) + [v]
                
            # Convert the source and sink candidates to their local indicies within the cluster
            for m1, candidates in source_and_sink_candidates.items():
                source_and_sink_candidates[m1] = [cluster.index(i) for i in candidates]

            self.generic_sub_problems.append(GenericSubProblem(scores = self.scores[cluster], 
                                                               positions = self.positions[cluster], 
                                                               source_and_sink_candidates = source_and_sink_candidates))
        
    def plot(self, show: bool = True):
        """Plots the sub problem graph with the edges between clusters shown"""
        plt.style.use("seaborn-v0_8-ticks")
        plt.gca().set_aspect("equal", adjustable="box")

        # Compute sizes for plotting
        min_score = min([score for score in self.scores if score != 0])
        max_score = max([score for score in self.scores])


        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:olive", "tab:gray", "tab:pink", "tab:purple", "tab:brown"] # TODO: add extra colors
        markers = ["o"]
        for sub_problem, (color, marker) in zip(self.generic_sub_problems, product(colors, markers)):
            plot_sub_problem(sub_problem, score_bounds = (min_score, max_score), color=color, marker=marker)

        # Plot edges between nodes in denulay triangulations
        already_plotted = set()
        for inter_cluster_edges_from_one_cluster in self.inter_cluster_edges.values():
            for edge in inter_cluster_edges_from_one_cluster:
                if edge in already_plotted:
                    continue

                points = self.positions[edge, :] 
                plt.plot(points[:, 0], points[:, 1], linestyle="dotted", color="black", zorder=0)
                already_plotted.add(tuple(reversed(edge)))
        
        if show:
            plt.show()

    def convert_to_concrete_sub_problems(self, cluster_route: List[List[int]]) -> List[ConcreteSubProblem]:
        """Converts the generic subproblems into concrete subproblems."""

if __name__ == "__main__":
    top_instance = TOPInstance.load_from_file("p4.2.d.txt")
    clusters = cluster_kmeans(top_instance, 8)
    g = GenericSubProblemGraph(top_instance, clusters)
    g.plot()