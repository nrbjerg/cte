from __future__ import annotations
from dataclasses import dataclass 
import numpy as np
from numpy.typing import ArrayLike
import os

@dataclass
class TOPInstance:
    """Stores the data related to a TOP instance."""
    problem_id: str
    number_of_agents: int
    t_max: float
    distance_matrix: ArrayLike
    score_array: ArrayLike 

    @staticmethod
    def load_from_problem_id(problem_file: str) -> TOPInstance:
        """Loads a TOP instance from a given problem id"""
        with open(os.path.join(os.getcwd(), "resources", "top", problem_file), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))[1:] # NOTE: skip the first line which contains information about the number of nodes.
            number_of_agents = int(lines[0].split(" ")[-1])
            t_max = float(lines[1].split(" ")[-1])

            positions, scores = [], []
            print(lines[2:])
            for (x_pos_str, y_pos_str, score_str) in map(lambda line: tuple(line.split("\t")), lines[2:]):
                positions.append((float(x_pos_str), float(y_pos_str)))
                scores.append(float(score_str))
            
            number_of_nodes = len(positions)

            distance_matrix = np.ones((number_of_nodes, number_of_nodes)) * np.inf
            for (i, (x_i, y_i)) in enumerate(positions):
                for (j, (x_j, y_j)) in enumerate(positions):
                    if i != j:
                        distance_matrix[i, j] = np.sqrt(np.square(x_i - x_j) + np.square(y_i - y_j))

            return TOPInstance(problem_file[:-4], number_of_agents, t_max, distance_matrix, np.array(scores)) 

if __name__ == "__main__":
    print(TOPInstance.load_from_problem_id("p4.2.a.txt"))
