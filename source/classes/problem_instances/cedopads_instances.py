# %% 
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from classes.data_types import Position, AreaOfAcquisition, Angle, State, compute_difference_between_angles, Vector
import dubins
import os 
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.lines as mlines
from library.core.dubins.relaxed_dubins import compute_relaxed_dubins_path, compute_length_of_relaxed_dubins_path
from numba import njit

plt.rcParams['figure.figsize'] = [9, 9]

# ------------------------------------------------- Utility Functions ------------------------------------------- #
UtilityFunction = Callable[[float, Angle, Angle, Angle, Angle, float, Angle], float]

#@njit()
def utility_fixed_optical(base_line_score: float, theta: Angle, phi: Angle, psi: Angle, tau: Angle, r: float, eta: Angle) -> float:
    """Used to compute the score given all of the values."""
    # The difference between psi and theta measured in radians.
    if (phi == 0 or eta == 0):
        print(f"{phi=}, {eta=}")
    if phi != 3.142:
        weight_from_psi = np.cos((np.pi * compute_difference_between_angles(psi, theta)) / (6 * phi))
    else:
        weight_from_psi = 1

    weight_from_tau = np.cos((np.pi * compute_difference_between_angles(tau, (psi + np.pi) % (2 * np.pi))) / (3 * eta))

    return base_line_score * weight_from_psi * weight_from_tau

# ------------------------------------------------- Misc Definitions -------------------------------------------- #

# Models a CEDOPADS route
Visit = Tuple[int, Angle, Angle, float]
CEDOPADSRoute = List[Visit] # a list of tuples of the form (k, psi, tau, r)

@njit()
def _compute_position(pos: Position, r: float, psi: float) -> Position:
    """Computes the position of the a state, on the angle interval of the node at the given position"""
    return pos + r * np.array([np.cos(psi), np.sin(psi)])

@dataclass()
class CEDOPADSNode:
    """Models a node in the DTOPADS problem."""
    node_id: int
    pos: Position 
    base_line_score: float 
    thetas: Vector 
    phis: Vector 

    AOAs: List[AreaOfAcquisition] = field(init = False)

    # Purely for plotting purposes
    size: float | None = field(init = False)

    def __post_init__(self):
        """Compute the AOAs from the given angles, note that we do not include the sensing radii"""
        self.AOAs = [AreaOfAcquisition(theta, phi) for theta, phi in zip(self.thetas, self.phis)]

    @staticmethod
    def load_from_line(line: str, node_id: int) -> CEDOPADSNode:
        """Loads a CEDOPADS node from a line"""
        position_and_score, angle_specification = line.split(" ")[:3], line.split(" ")[3:]
        x, y, score = map(float, position_and_score)
        pos = np.array((x, y))

        thetas, phis = [], []
        for idx in range(len(angle_specification) // 2):
            thetas.append(float(angle_specification[2 * idx]))
            phis.append(float(angle_specification[2 * idx + 1]))

        return CEDOPADSNode(node_id, pos, score, np.array(thetas), np.array(phis))
         
    def compute_score(self, psi: Angle, tau: Angle, r: float, eta: Angle, utility_function: UtilityFunction) -> float:
        """Computes the score of the given node, given a heading angle psi."""
        for j, aoa in enumerate(self.AOAs):
            if aoa.contains_angle(psi):
                return utility_function(self.base_line_score, self.thetas[j], self.phis[j], psi, tau, r, eta) 

        return 0 

    def get_AOA(self, psi: float) -> Optional[AreaOfAcquisition]:
        """Returns the angle interval which contains the given psi."""
        for aoa in self.AOAs:
            if aoa.contains_angle(psi): 
                return aoa
            
        # If there is no angle inteval which contains psi return none
        return None 

    def plot(self, r_min: float, r_max: float, color: str = "tab:gray", alpha = 0.2):
        """Plots the point, and the angle cones."""
        plt.scatter(self.pos[0], self.pos[1], self.size, c = color)
        
        for aoa in self.AOAs:
            aoa.plot(self.pos, r_min, r_max, color, color, alpha=alpha)

    def get_state(self, r: float, psi: Angle, tau: Angle) -> State:
        """Computes the state corresponding to a given visit."""
        return State(_compute_position(self.pos, r, psi), tau)
                    
@dataclass
class CEDOPADSInstance:
    """Stores the information related to an instance of the Dubins Team Orientering Problem with Angle Dependent Scores"""
    problem_id: str
    source: Position
    sink: Position
    nodes: List[CEDOPADSNode]
    sensing_radii: Tuple[float, float]
    t_max: float
    rho: float
    eta: Angle
    
    @staticmethod
    def load_from_file(file_name: str, needs_plotting: bool = False) -> CEDOPADSInstance:
        """Loads a DTOPADS instance from a given problem id."""
        #print(file_name)
        with open(os.path.join(os.getcwd(), "resources", "CEDOPADS", file_name), "r") as file:
            lines = list(map(lambda line: line.strip(), file.read().splitlines())) 
            t_max = float(lines[1].split(" ")[1])
            r_min = float(lines[2].split(" ")[1])
            r_max = float(lines[3].split(" ")[1])
            eta = float(lines[4].split(" ")[1])
            rho = float(lines[5].split(" ")[1])
            (x_pos, y_pos, _) = map(float, lines[6].split(" "))
            source = np.array((x_pos, y_pos))

            (x_pos, y_pos, _) = map(float, lines[-1].split(" "))
            sink = np.array((x_pos, y_pos))

            nodes = []
            for node_id, line in enumerate(lines[7:-1]):
                nodes.append(CEDOPADSNode.load_from_line(line, node_id))

            if needs_plotting:
                min_score = min([node.base_line_score for node in nodes if node.base_line_score != 0])
                max_score = max([node.base_line_score for node in nodes])

                node_sizes = [(0.2 + (node.base_line_score - min_score) / (max_score - min_score)) * 100 for node in nodes] # Normalized using min-max feature scaling
                for i, size in enumerate(node_sizes):
                    nodes[i].size = size

            return CEDOPADSInstance(file_name[:-4], source, sink, nodes, (r_min, r_max), t_max, rho, eta) 

    def plot(self, aoas_to_exclude: Dict[int, int], show: bool = False):
        """Displays a plot of the problem instance, and displays the coresponding angles, but excludeds the angle intervals specified by the angle_intervals_to_exclude dictionary containing node indicies and angle interval indicies."""
        plt.style.use("bmh")

        plt.scatter(*self.source, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 120, marker = "D", c = "black", zorder=4)

        # Plots targets
        for k, node in enumerate(self.nodes):
            if k not in aoas_to_exclude.keys():
                plt.scatter(*node.pos, node.size, c = "tab:gray")
        
            for i, aoa in enumerate(node.AOAs):
                if i != aoas_to_exclude.get(k, -1):
                    aoa.plot(node.pos, r_min = self.sensing_radii[0], r_max = self.sensing_radii[1], color = "tab:gray", alpha = 0.1)

        if show:
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show()

    def plot_with_route(self, route: CEDOPADSRoute, utility_function: UtilityFunction, color: str = "tab:orange", show: bool = False):
        """Plots the CEDOPADS instance with a route"""
        plt.style.use("bmh")
        used_angle_intervals = {} 
        for (k, psi, _, _) in route:
            for i, aoa in enumerate(self.nodes[k].AOAs):
                if aoa.contains_angle(psi):
                    used_angle_intervals[k] = i 

        self.plot(used_angle_intervals)

        if len(route) == 0:
            raise ValueError("Got an empty route.")
        
        else:
            q = self.get_states(route)
            for i, (k, _, _, _) in enumerate(route):
                self.nodes[k].AOAs[used_angle_intervals[k]].plot(self.nodes[k].pos, r_min = self.sensing_radii[0], r_max = self.sensing_radii[1], color = color, alpha = 0.2)
                #self.nodes[k].plot(sensing_radius, color = color)
                plt.scatter(*self.nodes[k].pos, c = color, s = self.nodes[k].size, zorder=2)
                plt.scatter(*q[i].pos, marker="s", c = color, zorder=2)

            # 1. Plot route from the source to q_1
            first_path = compute_relaxed_dubins_path(q[0].angle_complement(), self.source, self.rho)
            first_path.plot(q[0].angle_complement(), self.source, color = color)

            # 2. Plot the dubins trajectories between q_i, q_i + 1
            for i in range(len(q) - 1):
                if q[i] != q[i + 1]:
                    configurations = np.array(dubins.path_sample(q[i].to_tuple(), q[i + 1].to_tuple(), self.rho, 0.1)[0])
                    plt.plot(configurations[:, 0], configurations[:, 1], c = color)

            # 3. Plot route from q_M to the sink 
            final_path = compute_relaxed_dubins_path(q[-1], self.sink, self.rho)
            final_path.plot(q[-1], self.sink, color = color)

        label = f"$Q_I = {self.compute_score_of_route(route, utility_function):.2f}$, $D_I = {self.compute_length_of_route(route):.2f}$"
        indicators = [mlines.Line2D([], [], color=color, label=label, marker = "s")]
        plt.legend(handles=indicators, loc=1)
        plt.title(f"{self.problem_id}, ($t_{{max}}: {self.t_max}, \\rho: {self.rho}, \\eta: {self.eta}$)")

        if show:
            plt.plot()
    
    def get_state (self, visit: Visit) -> State:
        """Gets the state associated with the given visit"""
        return self.nodes[visit[0]].get_state(visit[3], visit[1], visit[2])

    def get_states(self, route: CEDOPADSRoute) -> List[State]:
        """Returns a list of states corresponding to q_1, ..., q_M from the problem formulation."""
        return [self.nodes[k].get_state(r, psi, tau) for k, psi, tau, r in route]

    def compute_length_of_route(self, route: CEDOPADSRoute) -> float:
        """Computes the length of the route, for the given sensing radius and turning radius rho."""
        return sum(self.compute_lengths_of_route_segments(route))

    def compute_lengths_of_route_segments(self, route: CEDOPADSRoute) -> List[float]:
        """Computes the length of the route, for the given sensing radius and turning radius rho."""
        if len(route) == 0:
            return [np.linalg.norm(self.source - self.sink)]

        q = self.get_states(route) 
        tups = [q[i].to_tuple() for i in range(len(q))]

        return ([compute_length_of_relaxed_dubins_path(q[0].angle_complement(), self.source, self.rho)] + 
                [dubins.shortest_path(tups[i], tups[i + 1], self.rho).path_length() for i in range(len(q) - 1) if tups[i] != tups[i + 1]] +
                [compute_length_of_relaxed_dubins_path(q[-1], self.sink, self.rho)])

    def compute_score_of_route(self, route: CEDOPADSRoute, utility_function: UtilityFunction) -> float:
        """Computes the score of a CEDOPADS route."""
        # Check if we visit the same node more than once, in which case 
        # we discard the lowest score, in order to incentivise the algorithms to only visit the same node once.
        #if len(set(k for k, _ , _ in route)) < len(route):
        #    scores = {}
        #    for (k, psi, tau) in route:
        #        if (score := self.nodes[k].compute_score(psi, tau, eta)) > scores.get(k, 0):
        #            scores[k] = score

        #    return sum(score for score in scores.values())
        
        return sum([self.compute_score_of_visit(visit, utility_function) for visit in route])

    def compute_distance_between_state_and_visit(self, q: State, visit: Visit) -> float:
        """Computes the distance between visit0 and visit1"""
        (k, psi, tau, r) = visit
        q_of_visit = self.nodes[k].get_state(r, psi, tau)

        return dubins.shortest_path(q.to_tuple(), q_of_visit.to_tuple(), self.rho).path_length()

    def compute_score_of_visit(self, visit: Visit, utility_function: UtilityFunction) -> float:
        """Computes the score of a visit."""
        k, psi, tau, r = visit
        return self.nodes[k].compute_score(psi, tau, r, self.eta, utility_function)

    def compute_scores_along_route(self, route: CEDOPADSRoute, utility_function: UtilityFunction) -> Dict[int, float]:
        """Computes the scores along the route and returns them in a dictionary, with the indicies as keys."""
        return {i: self.compute_score_of_visit(visit, utility_function) for i, visit in enumerate(route)}

    def is_route_feasable(self, route: CEDOPADSRoute) -> bool:
        """Checks if the route is feasable."""
        return self.compute_length_of_route(route) <= self.t_max
    
    def compute_sdr(self, visit: Visit, route: CEDOPADSRoute, idx: int, utility_function: UtilityFunction) -> float:
        """Computes the sdr score of inserting the visit into the route at index idx."""
        score = self.compute_score_of_visit(visit, utility_function)
        q = self.get_state(visit)
        if idx == 0:
            tups = (self.get_state(route[idx - 1]).to_tuple(), q.to_tuple())
            return score / (dubins.shortest_path(tups[0], tups[1], self.rho).path_length() + 
                            compute_length_of_relaxed_dubins_path(q.angle_complement(), self.source, self.rho))
        elif idx == len(route) - 1:
            tups = (self.get_state(route[idx - 1]).to_tuple(), q.to_tuple())
            return score / (dubins.shortest_path(tups[0], tups[1], self.rho).path_length() + 
                            compute_length_of_relaxed_dubins_path(q, self.sink, self.rho))
        else:
            tups = (self.get_state(route[idx - 1]).to_tuple(), q.to_tuple(), self.get_state(route[idx + 1]).to_tuple())
            return score / (dubins.shortest_path(tups[0], tups[1], self.rho).path_length() + 
                            dubins.shortest_path(tups[1], tups[2], self.rho).path_length())

def load_CEDOPADS_instances(needs_plotting: bool = False) -> List[CEDOPADSInstance]:
    """Loads the set of TOP instances saved within the resources folder."""
    folder_with_CEDOPADS_instances = os.path.join(os.getcwd(), "resources", "CEDOPADS")
    files = os.listdir(folder_with_CEDOPADS_instances)
    return [CEDOPADSInstance.load_from_file(file, needs_plotting = needs_plotting) for file in files]
            