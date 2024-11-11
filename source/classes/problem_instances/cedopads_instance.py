from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Tuple
from classes.data_types import Position, AngleInterval
import os 
import numpy as np
from matplotlib import pyplot as plt 

class CEDOPADSNode:
    """Models a node in the DTOPADS problem."""
    pos: Tuple[float, float]
    score: float 

    thetas: List[float]
    phis: List[float] 
    zetas: List[float]
    intervals: List[AngleInterval]

    # Purely for plotting purposes
    size: float | None

    def __init__(self, pos: Position, score: float, thetas: List[float], phis: List[float], zetas: List[float]):
        """Initializes the Node, and computes the relevant intervals"""
        self.pos = pos 
        self.score = score
        self.thetas = thetas
        self.phis = phis
        self.zetas = zetas
    
        self.intervals = [AngleInterval(theta - phi, theta + phi) for theta, phi in zip(thetas, phis)]

    @staticmethod
    def load_from_line(line: str) -> CEDOPADSNode:
        """Loads a CEDOPADS node from a line"""
        print(line)
        position_and_score, angle_part = tuple(line.split(":"))
        x, y, score = map(float, position_and_score.split(" "))
        pos = np.array((x, y))

        thetas, phis, zetas = [], [], []
        for part in angle_part.split(","):
            theta, phi, zeta = map(float, part.split(" "))
            thetas.append(theta)
            phis.append(phi)
            zetas.append(zeta)

        return CEDOPADSNode(pos, score, thetas, phis, zetas)
         
    def compute_score(self, psi: float) -> float:
        """Computes the score of the given node, given a heading angle psi."""
        for k, interval in zip(self.intervals):
            if interval.contains(psi):
                return self.score * np.sin((self.phis[k] - psi) / self.zetas[k])

        return 0 

    def plot(self, r: float, color: str = "tab:gray") -> None:
        """Plots the point, and the angle cones."""
        plt.scatter(self.pos[0], self.pos[1], self.size, c = color)
        
        for interval in self.intervals:
            interval.plot(self.pos, r, color, 0.3)
                    
@dataclass
class CEDOPADS:
    """Stores the information related to an instance of the Dubins Team Orientering Problem with Angle Dependent Scores"""
    problem_id: str
    source: Position
    sink: Position
    nodes: List[CEDOPADSNode]
    
    @staticmethod
    def load_from_file(file_name: str, needs_plotting: bool = False) -> CEDOPADS:
        """Loads a DTOPADS instance from a given problem id."""
        with open(os.path.join(os.getcwd(), "resources", "CEDOPADS", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines())) 
            tmax = float(lines[0].split(" ")[0])
            (x_pos, y_pos, _) = map(float, lines[1].split(" "))
            source = np.array((x_pos, y_pos))

            (x_pos, y_pos, _) = map(float, lines[2].split(" "))
            sink = np.array((x_pos, y_pos))

            nodes = []
            for line in lines[3:]:
                nodes.append(CEDOPADSNode.load_from_line(line))

            if needs_plotting:
                min_score = min([node.score for node in nodes if node.score != 0])
                max_score = max([node.score for node in nodes])

                node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes] # Normalized using min-max feature scaling
                for i, size in enumerate(node_sizes):
                    nodes[i].size = size

            for node in nodes:
                print(node.intervals)

            return CEDOPADS(file_name[:-4], source, sink, nodes) 

    def plot(self, r: float, show: bool = False):
        """Displays a plot of the problem instance, and displays the coresponding angles."""
        plt.style.use("bmh")

        plt.scatter(*self.source, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 120, marker = "D", c = "black", zorder=4)

        # Plots targets
        for node in self.nodes:
            node.plot(r)
        
        if show:
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show()
            
if __name__ == "__main__":
    problem = CEDOPADS.load_from_file("set_64_1_60.txt", needs_plotting = True)
    problem.plot(r  = 1, show = True)