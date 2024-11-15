# %% 
from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Tuple
from classes.data_types import Position, AngleInterval, Angle, State
import dubins
import os 
import numpy as np
from matplotlib import pyplot as plt 
from library.core.relaxed_dubins import compute_relaxed_dubins_path

plt.rcParams['figure.figsize'] = [9, 9]

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

            return CEDOPADS(file_name[:-4], source, sink, nodes) 

    def plot(self, sensing_radius: float, show: bool = False):
        """Displays a plot of the problem instance, and displays the coresponding angles."""
        plt.style.use("bmh")

        plt.scatter(*self.source, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 120, marker = "D", c = "black", zorder=4)

        # Plots targets
        for node in self.nodes:
            node.plot(sensing_radius)
        
        if show:
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show()

    def plot_with_route(self, route: List[Tuple[int, Angle]], sensing_radius: float, rho: float, color: str = "tab:orange"):
        """Plots the CEDOPADS instance with a route"""
        self.plot(sensing_radius)

        if len(route) == 0:
            raise ValueError("Got an empty route.")
        
        else:
            q = self.get_states(route, sensing_radius)
            for i, (k, _) in enumerate(route):
                plt.scatter(*self.nodes[k].pos, c = color, s = self.nodes[k].size, zorder=2)
                plt.scatter(*q[i].pos, marker="s", c = color, zorder=2)

            # 1. Plot route from the source to q_1
            k_1, psi_1 = route[0]
            q_tilde = State(q[0].pos, (np.pi + psi_1) % (2 * np.pi))
            first_path = compute_relaxed_dubins_path(q_tilde, self.source, rho, need_path=True)
            first_path.plot(q_tilde, self.source, color = color)

            # 2. Plot the dubins trajectories between q_i, q_i + 1
            for i in range(len(q) - 1):
                q0 = (q[i].pos[0], q[i].pos[1], q[i].angle)
                q1 = (q[i + 1].pos[0], q[i + 1].pos[1], q[i + 1].angle)
                configurations = np.array(dubins.path_sample(q0, q1, rho, 0.1)[0])
                plt.plot(configurations[:, 0], configurations[:, 1], c = color)

            # 3. Plot route from q_M to the sink 
            final_path = compute_relaxed_dubins_path(q[-1], self.sink, rho, need_path=True)
            final_path.plot(q[-1], self.sink, color = color)

        plt.plot()
    
    def get_states(self, route: List[Tuple[int, Angle]], sensing_radius: float) -> List[State]:
        """Returns a list of states corresponding to q_1, ..., q_M from the problem formulation."""
        states = []
        for k, psi in route:
            pos = self.nodes[k].pos - sensing_radius * np.array([np.cos(psi), np.sin(psi)])
            states.append(State(pos, psi))
        return states


    def compute_length_of_route(self, route: List[Tuple[int, Angle]], sensing_radius: float, rho: float) -> float:
        """Computes the length of the route, for the given sensing radius and turning radius rho."""
        q = self.get_states(route, sensing_radius) 
        tups = [(q[i].pos[0], q[i].pos[1], q[i].angle) for i in range(len(q))]

        q_tilde = State(q[0].pos, (np.pi + q[0].angle) % (2 * np.pi))

        return (compute_relaxed_dubins_path(q_tilde, self.source, rho) + 
                sum([dubins.shortest_path(tups[i], tups[i + 1], rho).path_length() for i in range(len(q) - 1)]) +
                compute_relaxed_dubins_path(q[-1], self.sink, rho))


    def is_route_feasable(self, route: List[Tuple[int, Angle]], sensing_radius: float, rho: float, tmax: float) -> bool:
        """Checks if the route is feasable."""
        return self.compute_length_of_route(route, sensing_radius, rho) <= tmax
            
if __name__ == "__main__":
    problem = CEDOPADS.load_from_file("set_64_1_60.txt", needs_plotting = True)
    route = [(25, np.pi / 2), (6, -np.pi / 4), (20, np.pi)]
    rho = 1.4
    sensing_radius = 0.6
    problem.plot_with_route(route, sensing_radius = sensing_radius, rho = rho) 
    print(problem.compute_length_of_route(route, sensing_radius, rho))