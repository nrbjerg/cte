# %%
from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict
from classes.data_types import Position, Matrix
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
from itertools import combinations
from scipy.spatial import Delaunay
from copy import deepcopy
 
# Source for this: https://www.moodle.aau.dk/pluginfile.php/3267197/mod_resource/content/1/as4.pdf 
class Orientation(Enum): 
    """Models the orientation of a list of 3 points."""
    Collinear = 0
    CounterClockwise = 1
    Clockwise = 2

    @staticmethod
    def compute_orientation(points: Tuple[Position, Position, Position]) -> Orientation:
        """Computes the orientation of points[2] wrt the linesegment from points[0] to points[1]."""
        match (points[1][1] - points[0][1]) * (points[2][0] - points[1][0]) - (points[2][1] - points[1][1]) * (points[1][0] - points[0][0]):
            case num if num < 0:
                return Orientation.CounterClockwise
            case num if num > 0:
                return Orientation.Clockwise
            case _:
                return Orientation.Collinear

# As sugested on: https://numpy.org/doc/2.1/reference/generated/numpy.cross.html
def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

Edge = Tuple[int, int]
Linesegment = Tuple[Position, Position]

# Source (page 1018, chapter 33): https://www.moodle.aau.dk/pluginfile.php/2832809/mod_page/content/7/Computational%20Geometry.pdf 
def _on_segment(seg: Linesegment, p: Position) -> bool:
    """Checks if the point p is on the linesegment, if we know that the orientation between the endpoints of seg and p are collinear."""
    if (p[0] > min(seg[0][0], seg[1][0]) and p[0] < max(seg[0][0], seg[1][0]) and
        p[1] > min(seg[0][1], seg[1][1]) and p[1] < max(seg[0][1], seg[1][1])):
        return True

    return False

# Source (page 1018, chapter 33): https://www.moodle.aau.dk/pluginfile.php/2832809/mod_page/content/7/Computational%20Geometry.pdf 
def _intersects (seg0: Linesegment, seg1: Linesegment)-> bool:
    """Checks if the two edges intersect each other, when the vertex is embedded into the plane using the given embeding."""

    # Check if the two edges intersect, by considering the orientations of each of the 
    # indient verticies, with respect to the other edge.
    orientations = (cross2d(seg1[0] - seg0[0], seg0[1] - seg0[0]),
                    cross2d(seg1[1] - seg0[0], seg0[1] - seg0[0]),
                    cross2d(seg0[0] - seg1[0], seg1[1] - seg1[0]),
                    cross2d(seg0[1] - seg1[0], seg1[1] - seg1[0]))

    # FIXME: the last line in the criteria might cause bugs.
    if ((((orientations[0] > 0 and orientations[1] < 0) or (orientations[0] < 0 and orientations[1] > 0)) and 
        ((orientations[2] > 0 and orientations[3] < 0) or (orientations[2] < 0 and orientations[3] > 0))) and 
        not (all(seg0[0] == seg1[0]) or all(seg0[0] == seg1[1]) or all(seg0[1] == seg1[0]) or all(seg0[1] == seg1[1]))): 
        return True
    elif orientations[0] == 0.0 and _on_segment(seg0, seg1[0]):
        return True
    elif orientations[1] == 0.0 and _on_segment(seg0, seg1[1]):
        return True
    elif orientations[2] == 0.0 and _on_segment(seg1, seg0[0]):
        return True
    elif orientations[3] == 0.0 and _on_segment(seg1, seg0[1]):
        return True
    else:
        return False


@dataclass 
class Graph:
    """Models a graph."""
    edges: List[Edge]
    adj_mat: Matrix
    n: int

    @staticmethod
    def initialize_random_graph(n: int) -> Graph:
        """Initializes a random graph, with n verticies."""
        # We start by computing a random adjacency matrix.
        A = np.random.choice(2, p = (0.6, 0.4), size = (n, n))
        adj_mat = np.ceil((A + A.T) / 2).astype("int64")

        # Make sure we are working with a simple graph.
        for v in range(n):
            adj_mat[v, v] = 0

        edges = sum([[(v, u) for v in range(u, n) if adj_mat[v, u] == 1] for u in range(n)], [])

        return Graph(edges, adj_mat, n)

    @staticmethod
    def initialize_random_close_to_planar_graph(n: int, p: float = 0.4) -> Graph:
        """Initializes a random planar graph."""
        positions = np.random.uniform(0, 1, size = (n, 2))

        # NOTE: A simplicity is simply an N-dimensional triangle
        triangulation = Delaunay(positions)
        adj_mat = np.zeros((n, n))
        for simplex in triangulation.simplices: 
            for i in simplex:
                for j in simplex:
                    if i == j:
                        continue
                    adj_mat[i][j] = 1
                    adj_mat[j][i] = 1
        
        for v in range(n):
            for u in range(v + 1, n):
                if adj_mat[v][u] == 0 and np.random.uniform(0, 1) < p:
                    adj_mat[v][u] = 1
                    adj_mat[u][v] = 1

        edges = sum([[(v, u) for v in range(u, n) if adj_mat[v, u] == 1] for u in range(n)], [])

        g = Graph(edges, adj_mat, n)
        g.plot(positions)
        print(f"Crossing Number of generated graph: {g.crossing_number(positions)}")
        return g

    def plot(self, embedding: Matrix):
        """Plots the embeding using matplotlib."""
        # Plot verticies at the given positions
        for v in range(self.n):
            plt.text(*embedding[v], str(v), fontsize=12, ha='center', va='center', color='white',
                 bbox=dict(facecolor='black', edgecolor='none', boxstyle='circle'), zorder=2)

        # Plot the edges of the graph, marking the intersecting ones with red.
        problem_edge_indicies = set()
        for e0, e1 in combinations(self.edges, 2):
            seg0 = (embedding[e0[0]], embedding[e0[1]])
            seg1 = (embedding[e1[0]], embedding[e1[1]])
            if _intersects(seg0, seg1):
                problem_edge_indicies.add(self.edges.index(e0))
                problem_edge_indicies.add(self.edges.index(e1))
        
        for idx, edge in enumerate(self.edges):
            xs = [embedding[edge[0]][0], embedding[edge[1]][0]]
            ys = [embedding[edge[0]][1], embedding[edge[1]][1]]
            if idx not in problem_edge_indicies:
                plt.plot(xs, ys, c = "tab:green", zorder = 1)
            else:
                plt.plot(xs, ys, c = "tab:red", zorder = 1)

        plt.xlim(np.min(embedding[:, 0]) - 0.1, np.max(embedding[:, 0]) + 0.1) 
        plt.ylim(np.min(embedding[:, 1]) - 0.1, np.max(embedding[:, 1]) + 0.1) 
        plt.show()

    def crossing_number(self, embedding: Matrix, debug: bool = False) -> int:
        """Computes the number of intersections betweeen edges, for a embedding which maps
           every vertex v in G to positions[v]"""
        intersection_count = 0
        for e0, e1 in combinations(self.edges, 2):
            seg0 = (embedding[e0[0]], embedding[e0[1]])
            seg1 = (embedding[e1[0]], embedding[e1[1]])
            if _intersects(seg0, seg1):
                intersection_count += 1
                if debug:
                    print(e0, e1)

        return intersection_count

    def number_of_adjacent_edges_which_cross_another_edge(self, embedding: Matrix) -> Dict[int, int]:
        """Computes the number of intersections betweeen edges, for a embedding which maps
           every vertex v in G to positions[v]"""
        count = {v: 0 for v in range(self.n)}
        for e0, e1 in combinations(self.edges, 2):
            seg0 = (embedding[e0[0]], embedding[e0[1]])
            seg1 = (embedding[e1[0]], embedding[e1[1]])
            if _intersects(seg0, seg1):
                count[e0[0]] += 1
                count[e0[1]] += 1
                count[e1[0]] += 1
                count[e1[1]] += 1

        return count
    
def genetic_algortihm(g: Graph, time_budget: float = 60.0, n_pop: int = 128) -> Matrix:
    """Runs a genetic algorithm for finding an embedding with a low crossing number."""
    starting_time = time.time()
    # 1. initialize population
    generation = [np.random.uniform(0, 1, size=(g.n, 2)) for _ in range(n_pop)]
    fitnesses = np.array([g.crossing_number(embedding) for embedding in generation])
    idx = fitnesses.argmin() 
    fitness_of_best_embedding = fitnesses[idx]
    best_embedding = generation[idx]

    # 2. Perform crossover ect.
    generation_number = 0
    while time.time() - starting_time < time_budget and fitness_of_best_embedding != 0:
        print(f"generation: {generation_number} best embedding has a crossing number of: {fitness_of_best_embedding}")

        rankings = np.argsort(fitnesses)
        modified_fitnesses = 1 - np.exp(-(rankings ** 2))
        probs = modified_fitnesses / np.sum(modified_fitnesses) 

        # Crossover & mutation.
        new_generation = []
        for i in range(n_pop):
            parent_indicies = np.random.choice(n_pop, size = 2, replace = False, p = probs)
            parents = [generation[i] for i in parent_indicies]
            indicies = np.random.choice(2, size=g.n, p = (0.8, 0.2))

            non_mutated_offspring = [np.array([parents[parent_index][i] for i, parent_index in enumerate(indicies)]), 
                                     np.array([parents[parent_index][1 - i] for i, parent_index in enumerate(indicies)])]
            
            # TODO: This mutation algorithm is shit, use truncated normal distributions instead.
            for child in non_mutated_offspring:
                seen = []
                # Uniform random mutation
                for j, mutate in enumerate(np.random.choice(2, size = g.n, p = (0.9, 0.1))):
                    if mutate == 1 or any(all(child[j] == allele) for allele in seen):
                        child[j] = np.random.uniform(0, 1, size = (1, 2))
                    seen.append(child[j])

                new_generation.append(child)
            
            # Add a random permutation to the parent at index i
            permutation = np.random.choice(g.n, size = g.n, replace = False)
            non_mutated_child = deepcopy(generation[i])
            new_generation.append(non_mutated_child[permutation])

            # Randomly mutate the worst points in the parent
            child = deepcopy(generation[i])
            count = g.number_of_adjacent_edges_which_cross_another_edge(child)
            for k, v in count.items():
                if v != 0 and np.random.uniform(0, 1) < 0.3:
                    child[k] = np.random.uniform(0, 1, size = (1, 2))


        # Compute fitnesses and select most fit individual in the new population
        fitnesses_of_new_generation = [g.crossing_number(embedding) for embedding in new_generation]
        indicies = np.argsort(fitnesses_of_new_generation)[:n_pop - 1]
        generation = [new_generation[i] for i in indicies]
        fitnesses = np.array([fitnesses_of_new_generation[i] for i in indicies])
        
        # Elitism
        generation.append(best_embedding)
        fitnesses = np.append(fitnesses, [[fitness_of_best_embedding]])
        print(f"Average embedding: {np.sum(fitnesses) / n_pop}")

        idx = fitnesses.argmin() 
        fitness_of_best_embedding = fitnesses[idx]
        best_embedding = generation[idx]
        computed_crossing_number = g.crossing_number(best_embedding)
        print(f"New best crossing number: {computed_crossing_number}")
        generation_number += 1

    g.crossing_number(best_embedding, debug = True)
    g.plot(best_embedding)
    return best_embedding

if __name__ == "__main__":
    np.random.seed(0)
    n = 12
    start = time.time()
    g = Graph.initialize_random_close_to_planar_graph(n, p = 0.1)
    best_embedding = genetic_algortihm(g, time_budget = 180, n_pop = 256) 