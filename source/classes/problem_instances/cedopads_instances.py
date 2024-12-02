# %% 
from __future__ import annotations
from dataclasses import dataclass 
from typing import List, Tuple, Set, Optional
from classes.data_types import Position, AngleInterval, Angle, State, compute_difference_between_angles
import dubins
import os 
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.lines as mlines
from library.core.relaxed_dubins import compute_relaxed_dubins_path, compute_length_of_relaxed_dubins_path

plt.rcParams['figure.figsize'] = [9, 9]

# Models a CEDOPADS route
CEDOPADSRoute = List[Tuple[int, Angle, Angle]] # a list of tuples of the form (k, psi, tau)

class CEDOPADSNode:
    """Models a node in the DTOPADS problem."""
    node_id: int
    pos: Tuple[float, float]
    score: float 

    thetas: List[float]
    phis: List[float] 
    zetas: List[float]
    intervals: List[AngleInterval]

    # Purely for plotting purposes
    size: float | None

    def __init__(self, node_id: int, pos: Position, score: float, thetas: List[float], phis: List[float], zetas: List[float]):
        """Initializes the Node, and computes the relevant intervals"""
        self.node_id = node_id
        self.pos = pos 
        self.score = score
        self.thetas = thetas
        self.phis = phis
        self.zetas = zetas
    
        self.intervals = [AngleInterval(theta - phi, theta + phi) for theta, phi in zip(thetas, phis)]

    @staticmethod
    def load_from_line(line: str, node_id: int) -> CEDOPADSNode:
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

        #print(phis, zetas)
        assert all(phi <= zeta for phi, zeta in zip(phis, zetas))

        return CEDOPADSNode(node_id, pos, score, thetas, phis, zetas)
         
    def compute_score(self, psi: Angle, tau: Angle, eta: float) -> float:
        """Computes the score of the given node, given a heading angle psi."""
        for j, interval in enumerate(self.intervals):
            if interval.contains(psi):
                # The difference between psi and theta measured in radians.
                weight_from_psi = np.cos(compute_difference_between_angles(psi, self.thetas[j]) / self.zetas[j])
                weight_from_tau = np.exp(- compute_difference_between_angles(tau, (psi + np.pi) % (2 * np.pi)) / eta)
                #weight_from_tau = np.cos(eta * compute_difference_between_angles(tau, (psi + np.pi) % (2 * np.pi)))

                if weight_from_psi < 0 or weight_from_tau < 0:
                    self.plot(1)
                    q = self.get_state(1, psi, tau)
                    q.plot()
                    fov = AngleInterval(tau - eta / 2, tau + eta / 2)
                    fov.plot(q.pos, 1, "tab:orange", 0.1)
                    plt.show()
                    print(f"psi and theta {compute_difference_between_angles(psi, self.thetas[j])}, tau and psi {compute_difference_between_angles(tau, (psi + np.pi) % (2 * np.pi))}")
                    raise ValueError(f"one of the weights {weight_from_psi:.2f}, {weight_from_tau:.2f} is negative, psi: {psi:.2f}, theta: {self.thetas[j]:.2f}, phi: {self.phis[j]:.2f}, zeta: {self.zetas[j]:.2f}, tau: {tau:.2f}, eta: {eta:.2f}")

                return self.score * weight_from_psi * weight_from_tau 
            #np.cos((self.thetas[i] - psi) / self.zetas[i])

        return 0 

    def get_angle_interval(self, psi: float) -> Optional[AngleInterval]:
        """Returns the angle interval which contains the given psi."""
        for interval in self.intervals:
            if interval.contains(psi): 
                return interval
            
        # If there is no angle inteval which contains psi return none
        return None 

    def plot(self, sensing_radius: float, color: str = "tab:gray") -> None:
        """Plots the point, and the angle cones."""
        plt.scatter(self.pos[0], self.pos[1], self.size, c = color)
        
        for interval in self.intervals:
            interval.plot(self.pos, sensing_radius, color, 0.2)

        # Added for debuging
        #for theta in self.thetas:
        #    offset = sensing_radius * np.array([np.cos(theta), np.sin(theta)])
        #    plt.plot([self.pos[0], self.pos[0] - offset[0]], [self.pos[1], self.pos[1] - offset[1]], c = color)

    def get_state(self, sensing_radius: float, psi: Angle, tau: Angle) -> State:
        """Computes the state corresponding to """
        offset = sensing_radius * np.array([np.cos(psi), np.sin(psi)])
        return State(self.pos + offset, tau)
                    
@dataclass
class CEDOPADSInstance:
    """Stores the information related to an instance of the Dubins Team Orientering Problem with Angle Dependent Scores"""
    problem_id: str
    source: Position
    sink: Position
    nodes: List[CEDOPADSNode]
    
    @staticmethod
    def load_from_file(file_name: str, needs_plotting: bool = False) -> CEDOPADS:
        """Loads a DTOPADS instance from a given problem id."""
        #print(file_name)
        with open(os.path.join(os.getcwd(), "resources", "CEDOPADS", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines())) 
            (x_pos, y_pos, _) = map(float, lines[1].split(" "))
            source = np.array((x_pos, y_pos))

            (x_pos, y_pos, _) = map(float, lines[2].split(" "))
            sink = np.array((x_pos, y_pos))

            nodes = []
            for node_id, line in enumerate(lines[3:]):
                nodes.append(CEDOPADSNode.load_from_line(line, node_id))

            if needs_plotting:
                min_score = min([node.score for node in nodes if node.score != 0])
                max_score = max([node.score for node in nodes])

                node_sizes = [(0.2 + (node.score - min_score) / (max_score - min_score)) * 100 for node in nodes] # Normalized using min-max feature scaling
                for i, size in enumerate(node_sizes):
                    nodes[i].size = size

            return CEDOPADSInstance(file_name[:-4], source, sink, nodes) 

    def plot(self, sensing_radius: float, nodes_to_exculude: Set[int], show: bool = False):
        """Displays a plot of the problem instance, and displays the coresponding angles."""
        plt.style.use("bmh")

        plt.scatter(*self.source, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 120, marker = "D", c = "black", zorder=4)

        # Plots targets
        for k in set(range(len(self.nodes))).difference(nodes_to_exculude):
            self.nodes[k].plot(sensing_radius)
        
        if show:
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show()

    def plot_with_route(self, route: CEDOPADSRoute, sensing_radius: float, rho: float, eta: float, color: str = "tab:orange"):
        """Plots the CEDOPADS instance with a route"""
        self.plot(sensing_radius, set(k for k, _, _ in route))

        if len(route) == 0:
            raise ValueError("Got an empty route.")
        
        else:
            q = self.get_states(route, sensing_radius)
            for i, (k, _, _) in enumerate(route):
                self.nodes[k].plot(sensing_radius, color = color)
                #plt.scatter(*self.nodes[k].pos, c = color, s = self.nodes[k].size, zorder=2)
                plt.scatter(*q[i].pos, marker="s", c = color, zorder=2)

            # 1. Plot route from the source to q_1
            first_path = compute_relaxed_dubins_path(q[0].angle_complement(), self.source, rho)
            first_path.plot(q[0].angle_complement(), self.source, color = color)

            # 2. Plot the dubins trajectories between q_i, q_i + 1
            for i in range(len(q) - 1):
                q0 = (q[i].pos[0], q[i].pos[1], q[i].angle)
                q1 = (q[i + 1].pos[0], q[i + 1].pos[1], q[i + 1].angle)
                configurations = np.array(dubins.path_sample(q0, q1, rho, 0.1)[0])
                plt.plot(configurations[:, 0], configurations[:, 1], c = color)

            # 3. Plot route from q_M to the sink 
            final_path = compute_relaxed_dubins_path(q[-1], self.sink, rho)
            final_path.plot(q[-1], self.sink, color = color)

        indicators = [mlines.Line2D([], [], color=color, label=f"Score: {round(self.compute_score_of_route(route, eta), 2)}, Length: {round(self.compute_length_of_route(route, sensing_radius, rho), 2)}", marker="s")]
        plt.legend(handles=indicators, loc=1)
        plt.plot()
    
    def plot_with_routes(self, routes: List[CEDOPADSRoute], sensing_radius: float, rho: float, colors: List[str] = ["tab:orange", "tab:green", "tab:red", "tab:blue", "tab:purple", "tab:cyan"], show: bool = False):
        """Plots multiple CEDOPADS routes on top of the plot"""
        self.plot(sensing_radius, set())

        if len(routes) > len(colors):
            raise ValueError("Got to many routes compared to the number of colors.")

        for route, color in zip(routes, colors):
            q = self.get_states(route, sensing_radius)
            for q_i in q:
                plt.scatter(*q_i.pos, marker="s", c = color, zorder=2)

            first_path = compute_relaxed_dubins_path(q[0].angle_complement(), self.source, rho)
            first_path.plot(q[0].angle_complement(), self.source, color = color)

            # 2. Plot the dubins trajectories between q_i, q_i + 1
            for i in range(len(q) - 1):
                configurations = np.array(dubins.path_sample(q[i].to_tuple(), q[i + 1].to_tuple(), rho, 0.1)[0])
                plt.plot(configurations[:, 0], configurations[:, 1], c = color)

            # 3. Plot route from q_M to the sink 
            final_path = compute_relaxed_dubins_path(q[-1], self.sink, rho)
            final_path.plot(q[-1], self.sink, color = color)

        indicators = [mlines.Line2D([], [], color=color, label=f"Score: {round(self.compute_score_of_route(route), 2)}, Length: {round(self.compute_length_of_route(route, sensing_radius, rho), 2)}", marker="s") for color, route in zip(colors, routes)]
        plt.legend(handles=indicators, loc=1)
        if show: 
            plt.plot()

    def get_states(self, route: CEDOPADSRoute, sensing_radius: float) -> List[State]:
        """Returns a list of states corresponding to q_1, ..., q_M from the problem formulation."""
        if len(route) == 0:
            return [] 
        else:
            return [self.nodes[k].get_state(sensing_radius, psi, tau) for k, psi, tau in route]

    def compute_length_of_route(self, route: CEDOPADSRoute, sensing_radius: float, rho: float) -> float:
        """Computes the length of the route, for the given sensing radius and turning radius rho."""
        if len(route) == 0:
            return np.linalg.norm(self.source - self.sink)

        q = self.get_states(route, sensing_radius) 
        tups = [q[i].to_tuple() for i in range(len(q))]

        return (compute_length_of_relaxed_dubins_path(q[0].angle_complement(), self.source, rho) + 
                sum([dubins.shortest_path(tups[i], tups[i + 1], rho).path_length() for i in range(len(q) - 1)]) +
                compute_length_of_relaxed_dubins_path(q[-1], self.sink, rho))

    def compute_score_of_route(self, route: CEDOPADSRoute, eta: float) -> float:
        """Computes the score of a CEDOPADS route."""
        # Check if we visit the same node more than once, in which case 
        # we discard the lowest score, in order to incentivise the algorithms to only visit the same node once.
        if len(set(k for k, _ , _ in route)) < len(route):
            scores = {}
            for (k, psi, tau) in route:
                if (score := self.nodes[k].compute_score(psi, tau, eta)) > scores.get(k, 0):
                    scores[k] = score
            
            return sum(score for score in scores.values())
        
        return sum([self.nodes[k].compute_score(psi, tau, eta) for (k, psi, tau) in route])

    def is_route_feasable(self, route: CEDOPADSRoute, sensing_radius: float, rho: float, tmax: float) -> bool:
        """Checks if the route is feasable."""
        return self.compute_length_of_route(route, sensing_radius, rho) <= tmax

def load_CEDOPADS_instances(needs_plotting: bool = False) -> List[CEDOPADSInstance]:
    """Loads the set of TOP instances saved within the resources folder."""
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "CEDOPADS")
    return [CEDOPADSInstance.load_from_file(file_name, needs_plotting = needs_plotting) 
            for file_name in os.listdir(folder_with_top_instances)]
            
if __name__ == "__main__":
    #problem = CEDOPADSInstance.load_from_file("set_64_1_60.txt", needs_plotting = True)
    #route = [(25, np.pi / 2), (6, -np.pi / 4), (20, np.pi)]
    #rho = 1.4
    #sensing_radius = 0.6
    #problem.plot_with_route(route, sensing_radius = sensing_radius, rho = rho) 
    #print(problem.compute_length_of_route(route, sensing_radius, rho))

    line = "0 0 7.0:6.23 1.25 1.26"
    node = CEDOPADSNode.load_from_line(line, 0)
    node.size = 1
    eta = np.pi / 3
    for _ in range(100):
        psi = node.intervals[0].generate_uniform_angle()
        tau = (psi + np.pi + np.random.uniform( - eta / 2, eta / 2)) % (2 * np.pi)
        fov = AngleInterval(tau - eta / 2, tau + eta / 2)
        q = node.get_state(1, psi, tau)
        q.plot()
        fov.plot(q.pos, 1, "tab:orange", 0.1) 
        if node.compute_score(psi, tau, eta) < 0:
            raise ValueError("Got negative score")
    node.plot(1)
    plt.show()
    #node.compute_score(psi, 1.29, 1)

# %%
