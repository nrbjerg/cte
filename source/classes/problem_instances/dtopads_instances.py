from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Callable, Tuple
from classes.data_types import Angle 
import os 
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt 
from matplotlib import patches
import re

@dataclass
class DTOPADSNode:
    """Models a node in the DTOPADS problem."""
    pos: Tuple[float, float]
    phis: List[float] | None
    alphas: List[float] | None
    scores: List[float] | None

    def compute_score(self, theta: float, k: int) -> float:
        """Computes the score of the given node, given an angle theta."""
        if k >= len(self.phis):
            raise ValueError(f"Index: {k} is out of range")

        return np.cos(self.alphas[k] * (theta - self.phis[k])) * self.scores[k]

    def plot(self, color: str = "black", highest_score: float = None) -> None:
        """Plots the point, and the angle cones."""
        plt.scatter(self.pos[0], self.pos[1], c = color)
        
        if self.phis and self.alphas and self.scores:
            for phi, alpha, score in zip(self.phis, self.alphas, self.scores):
                angles = np.linspace(phi - np.pi / (2 * alpha), phi + np.pi / (2 * alpha), 50) # TODO: add points

                if highest_score:
                    scalar = 2 * score / highest_score
                else:
                    scalar = 2

                points_on_arc = np.vstack((self.pos[0] + scalar * np.cos(angles),  # SOMETHING IS GOING ON HERE
                                           self.pos[1] + scalar * np.sin(angles)))
                points = np.vstack([points_on_arc.T, self.pos])

                cone = patches.Polygon(points, closed=True)
                plt.gca().add_patch(cone)
        
@dataclass
class DTOPADS:
    """Stores the information related to an instance of the Dubins Team Orientering Problem with Angle Dependent Scores"""
    problem_id: str
    number_of_agents: int
    t_max: float
    source: DTOPADSNode
    sink: DTOPADSNode
    nodes: List[DTOPADSNode]
    highest_score: float | None
    
    @staticmethod
    def load_from_file(file_name: str, needs_plotting: bool = False) -> DTOPADS:
        """Loads a DTOPADS instance from a given problem id."""
        with open(os.path.join(os.getcwd(), "resources", "DTOPADS", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines()))[1:] # NOTE: skip the first line which contains information about the number of nodes.
            number_of_agents = int(lines[0].split(" ")[-1])
            t_max = float(lines[1].split(" ")[-1])

            # Load source and sink
            (x_pos, y_pos, _) = map(float, lines[2].split(" "))
            source = DTOPADSNode((x_pos, y_pos), None, None, None)

            (x_pos, y_pos, _) = map(float, lines[-1].split(" "))
            sink = DTOPADSNode((x_pos, y_pos), None, None, None)

            nodes = [source, sink]
            for idx, (x_pos, y_pos) in enumerate(map(lambda line: tuple(map(float, re.sub(r"\[.*\]", "", line).split(" ")[:-1])), lines[3:-1])):
                # NOTE: PLEASE DO NOT TOUCH THE FOLLOWING LINE
                ads_part = list(map(float, re.findall(r"\[.*\]", lines[idx + 3])[0].split(" ")[1:-1]))
                phis = ads_part[0::3]
                alphas = ads_part[1::3]
                scores = ads_part[2::3]
                nodes.append(DTOPADSNode((x_pos, y_pos), phis, alphas, scores))
                                         
            highest_score = max(sum(node.scores) for node in nodes if node.scores)
            
            #colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple, tab:brown", "tab:pink"] # TODO: add extra colors
            return DTOPADS(file_name[:-4], number_of_agents, t_max, source, sink, nodes, highest_score) 

    def plot(self, show: bool = True):
        """Displays a plot of the problem instance, and displays the coresponding angles."""
        plt.style.use("bmh")
        fig, ax = plt.subplots()

        for node in self.nodes:
            node.plot(highest_score=self.highest_score)
        
        if show:
            ax.set_aspect("equal", adjustable="box")
            plt.show()
            
if __name__ == "__main__":
    problem = DTOPADS.load_from_file("p4.3.i.txt")
    problem.plot()
    plt.show()