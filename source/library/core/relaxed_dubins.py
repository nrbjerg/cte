#%%
from classes.data_types import Position, State, Angle, Matrix
import matplotlib.patches
import numpy as np 
from matplotlib import pyplot as plt 
from typing import Union, Tuple
from enum import IntEnum
from dataclasses import dataclass
import matplotlib
from utils.geometry import get_intersection_points_between_circles

class Direction(IntEnum):
    """Models a direction of a turn"""
    RIGHT = 0
    LEFT = 1

@dataclass
class CSPath:
    """Models a path starting with a circular arc and ending with a straight path."""
    direction: Direction
    radians_traversed_on_arc: Angle
    rho: float
    length: float

    def plot(self, q: State, p: Position, color: str = "tab:orange"): 
        """Plots the path using matplotlib, using q for the inital state of the dubins vehicle."""
        #plt.style.use("bmh")
        #plt.scatter(*q.pos, marker = "s", c = color, zorder=2)
        #plt.scatter(*p, marker = "s", c = color, zorder=2)
        # 1. Plot the arc
        rotation_angle = np.pi / 2 - q.angle
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], 
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])

        if self.direction == Direction.RIGHT:
            center = rotation_matrix.T.dot(np.array([self.rho, 0])) + q.pos
            start, end = np.rad2deg(q.angle + np.pi / 2 - self.radians_traversed_on_arc), np.rad2deg(q.angle + np.pi / 2)

        else:
            center = rotation_matrix.T.dot(np.array([-self.rho, 0])) + q.pos
            start, end = np.rad2deg(q.angle - np.pi / 2), np.rad2deg(q.angle - np.pi / 2 + self.radians_traversed_on_arc)

        arc = matplotlib.patches.Arc(center, 2 * self.rho, 2 * self.rho, theta1=start, theta2=end, edgecolor=color, lw = 2, zorder=2)
        plt.gca().add_patch(arc)

        # 2. Plot the staight path
        if self.direction == Direction.RIGHT:
            theta = np.pi - self.radians_traversed_on_arc
        else:
            theta = self.radians_traversed_on_arc

        inflection_point = center + rotation_matrix.T.dot(self.rho * np.array([np.cos(theta), np.sin(theta)]))
        plt.plot([inflection_point[0], p[0]], [inflection_point[1], p[1]], c = color, zorder=2)
    
    def __repr__(self) -> str:
        """Returns a string representation of the path."""
        path_type = "LS" if self.direction == Direction.LEFT else "RS"
        return f"{path_type}(radians: {self.radians_traversed_on_arc:.2f}, length: {self.length:.2f})"

@dataclass
class CCPath:
    """Models a path starting with a circular arc and ending in a circular arc."""
    directions: Tuple[Direction, Direction]
    radians_traversed_on_arcs: Tuple[float, float]
    rho: float
    length: float

    def plot(self, q: State, p: Position, color: str = "tab:orange"):
        """Plots the path using matplotlib, using q for the inital state of the dubins vehicle."""
        #plt.style.use("bmh")
        #plt.scatter(*q.pos, marker = "s", c = color, zorder=2)
        #plt.scatter(*p, marker = "s", c = color, zorder=2)
        rotation_angle = np.pi / 2 - q.angle
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], 
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])

        (u, v) = self.radians_traversed_on_arcs
        # Plot the first arc.
        if self.directions[0] == Direction.RIGHT:
            injuction_angle = 90 + np.rad2deg(q.angle - u)
            first_center = rotation_matrix.T.dot(np.array([self.rho, 0])) + q.pos
            first_arc = matplotlib.patches.Arc(first_center, 2 * self.rho, 2 * self.rho,
                                           theta1=injuction_angle, theta2=np.rad2deg(q.angle) + 90, edgecolor=color, lw=2, zorder=2)
        else:
            injuction_angle = np.rad2deg(q.angle + u) - 90
            first_center = rotation_matrix.T.dot(np.array([-self.rho, 0])) + q.pos
            first_arc = matplotlib.patches.Arc(first_center, 2 * self.rho, 2 * self.rho,
                                           theta1=np.rad2deg(q.angle) - 90, theta2=injuction_angle, edgecolor=color, lw=2, zorder=2)
        plt.gca().add_patch(first_arc)

        # Plot the second arc.
        if self.directions[1] == Direction.LEFT:
            second_center = rotation_matrix.T.dot(np.array([self.rho * (1 - 2 * np.cos(u)), 2 * self.rho * np.sin(u)])) + q.pos
            second_arc = matplotlib.patches.Arc(second_center, 2 * self.rho, 2 * self.rho,
                                           theta1=injuction_angle + 180, theta2=injuction_angle + 180 + np.rad2deg(v), edgecolor=color, lw=2, zorder=2)
        else:
            second_center = rotation_matrix.T.dot(np.array([self.rho * (2 * np.cos(u) - 1), 2 * self.rho * np.sin(u)])) + q.pos
            second_arc = matplotlib.patches.Arc(second_center, 2 * self.rho, 2 * self.rho,
                                           theta1=180 - np.rad2deg(v) + injuction_angle, theta2=180 + injuction_angle, edgecolor=color, lw=2, zorder=2)

        plt.gca().add_patch(second_arc)

    def __repr__(self) -> str:
        """Returns a string representation of the path."""
        path_type = "RL" if self.directions == (Direction.RIGHT, Direction.LEFT) else "LR"
        return f"{path_type}(radians: {self.radians_traversed_on_arcs[0]:.2f} and {self.radians_traversed_on_arcs[1]:.2f}, length: {self.length:.2f})"

def compute_relaxed_dubins_path(q: State, p: Position, rho: float, plot: bool = False, verbose_plot: bool = False, need_path: bool = False) -> Union[CSPath, CCPath, float]:
    """Either computes the relaxed dubins path between the configuration q and the position p, or the length 
       of the relaxed dubins path between q and the position p, if need_path == False"""
    # 1. Transform the cordiante system so that q.pos is the origin and so that q.angle == pi / 2
    rotation_angle = np.pi / 2 - q.angle
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], 
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    p_prime = rotation_matrix.dot(p - q.pos) 
    # NOTE: Remember that this means that the q is located at 0, 0 with an initial heading of pi / 2

    c_l, c_r = np.array([-rho, 0]), np.array([rho, 0])

    if plot and verbose_plot:
        plt.gca().arrow(q.pos[0], q.pos[1], np.cos(q.angle), np.sin(q.angle), color = "tab:orange", zorder=2, head_width=0.05)
        for color, center in zip(["tab:blue", "tab:green"], [c_l, c_r]): 
            plt.scatter(*rotation_matrix.T.dot(center) + q.pos, c = color)
            plt.gca().add_patch(plt.Circle(rotation_matrix.T.dot(center) + q.pos, rho, color = color, fill = False))

    if p_prime[0] >= 0: # Candidates are LS and RL
        distance_from_c_r = np.linalg.norm(p_prime - c_r)
        if distance_from_c_r < rho or (distance_from_c_r == rho and p_prime[0] < 0):
            intersection_points = [point for point in get_intersection_points_between_circles(c_l, 2 * rho, p_prime, rho) if point[1] >= 0]
            second_arc_center = intersection_points[0]
            u = np.arctan(second_arc_center[1] / (rho + second_arc_center[0]))

            # v is the angle between the vector form the new c_l to c_r and the vector form c_l to p
            v = 2 * np.pi - np.arccos(np.dot(second_arc_center - c_l, second_arc_center - p_prime) / (2 * rho ** 2))
            length = rho * (u + v)
            if not need_path:
                return length
            
            path = CCPath((Direction.LEFT, Direction.RIGHT), (u, v), rho, rho * (u + v))

        else:
            #if p[1] >= 0:
            hyp = np.linalg.norm(c_r - p_prime)
            a = np.arccos(rho / hyp)
            if p_prime[1] >= 0:
                b = np.arccos(np.dot(p_prime - c_r, - c_r) / (rho * hyp)) - a
            else:
                b = 2 * np.pi - np.arccos(np.dot(p_prime - c_r, - c_r) / (rho * hyp)) - a 
            
            length = np.sqrt(hyp ** 2 - rho ** 2) + b * rho 

            if not need_path:
                return length
            
            path = CSPath(Direction.RIGHT, b, rho, length)

    else: 
        distance_from_c_l = np.linalg.norm(p_prime - c_l)
        if distance_from_c_l < rho or (distance_from_c_l == rho and p_prime[0] < 0):
            intersection_points = [point for point in get_intersection_points_between_circles(c_r, 2 * rho, p_prime, rho) if point[1] >= 0]
            second_arc_center = intersection_points[0]
            plt.plot(*second_arc_center, c="red", zorder=4)
            u = np.abs(np.arctan(second_arc_center[1] / (second_arc_center[0] - rho)))

            # v is the angle between the vector form the center of the second arc and to c_r and the vector form the center of the second arc to p
            v = 2 * np.pi - np.arccos(np.dot(second_arc_center - c_r, second_arc_center - p_prime) / (2 * rho ** 2))
            length = rho * (u + v)
            if not need_path:
                return length
            
            path = CCPath((Direction.RIGHT, Direction.LEFT), (u, v), rho, rho * (u + v))

        else:
            hyp = np.linalg.norm(c_l - p_prime)
            a = np.arccos(rho / hyp)
            if p_prime[1] >= 0:
                b = np.arccos(np.dot(p_prime - c_l, - c_l) / (rho * hyp)) - a
            else:
                b = 2 * np.pi - np.arccos(np.dot(p_prime - c_l, - c_l) / (rho * hyp)) - a 

            length = np.sqrt(hyp ** 2 - rho ** 2) + b * rho 

            if not need_path:
                return length
            
            path = CSPath(Direction.LEFT, b, rho, length)

    if plot:
        path.plot(q, p)
        
    return path

def can_be_reached(q: State, p: Position, rho: float, remaining_travel_budget: float) -> bool:
    """Simply checks if the position p can be reached using a relaxed dubins path from the state q."""
    return compute_relaxed_dubins_path(q, p, rho, need_path=False) < remaining_travel_budget 

if __name__ == "__main__":
    angle = np.pi / 3 
    q = State(np.array([-2.0, -5.6]), 4.71)
    p = np.array([0, -7])
    print(compute_relaxed_dubins_path(q, p, 1, plot = True, verbose_plot = False, need_path = True))