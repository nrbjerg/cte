"""This module provides an API for working with the dubins library, to make it easier to work with."""
import numpy as np
from typing import Iterable, Tuple, List
from classes.data_types import State, Dir
from copy import deepcopy
import matplotlib.pyplot as plt 
import matplotlib as mpl
import dubins

def plot_path(q: State, segment_types: Iterable[Dir], segment_lengths: Iterable[float], rho: float, color: str = "tab:green", linewidth: int = 1, linestyle: str = "-"):
    """Plots an arbitary dubins path, of the given segment types and parameters"""
    pos = deepcopy(q.pos)
    print(f"{pos=}")
    angle = deepcopy(q.angle)
    for typ, length in zip(segment_types, segment_lengths):
        match typ:
            case Dir.S:
                plt.plot([pos[0], pos[0] + np.cos(angle) * length],
                         [pos[1], pos[1] + np.sin(angle) * length], color = color, zorder=2, linewidth = linewidth, linestyle = linestyle)
                pos = pos + length * np.array([np.cos(angle), np.sin(angle)])
                print(f"{pos=}")

            case Dir.R:
                center = pos + rho * np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])
                arc = mpl.patches.Arc(center, 2 * rho, 2 * rho, theta2 = np.rad2deg(angle + np.pi / 2), theta1 = np.rad2deg(angle + np.pi / 2 - length / rho), edgecolor = color, zorder = 2, linewidth = linewidth, linestyle=linestyle)
                plt.gca().add_patch(arc) 
                pos = center + rho * np.array([np.cos(angle + np.pi / 2 - length / rho), np.sin(angle + np.pi / 2 - length / rho)])
                angle = (angle - length / rho) % (2 * np.pi)

            case Dir.L:
                center = pos + rho * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
                arc = mpl.patches.Arc(center, 2 * rho, 2 * rho, theta1 = np.rad2deg(angle - np.pi / 2), theta2 = np.rad2deg(angle - np.pi / 2 + length / rho), edgecolor = color, zorder = 2, linewidth = linewidth, linestyle=linestyle)
                plt.gca().add_patch(arc) 
                pos = center + rho * np.array([np.cos(angle - np.pi / 2 + length / rho), np.sin(angle - np.pi / 2 + length / rho)])
                angle = (angle + length / rho) % (2 * np.pi)

def sample_dubins_path(q_i: State, q_f: State, rho: float, delta: float) -> List[State]:
    """Samples the dubins path from q_i to q_f, every delta units."""
    states = [q_i]

    pos = deepcopy(q_i.pos)
    angle = deepcopy(q_i.angle)
    segment_types, segment_lengths = call_dubins(q_i, q_f, rho)
    eps = 0.01
    for typ, length in zip(segment_types, segment_lengths):
        match typ:
            case Dir.S:
                n = int(np.ceil(length / delta))
                for k in range(n + 1):
                    states.append(State(np.array([pos[0] + np.cos(angle) * length * (k / n), pos[1] + np.sin(angle) * length * (k / n)]), angle))

                # Update position
                pos += length * np.array([np.cos(angle), np.sin(angle)])

            case Dir.R:
                center = pos + rho * np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])

                n = int(np.ceil(length / delta))
                for k in range(1, n):
                    angle_of_state = (angle - (length / rho) * (k / n)) % (2 * np.pi)
                    pos_of_state = center + (rho + eps) * np.array([np.cos(angle_of_state + np.pi / 2), np.sin(angle_of_state + np.pi / 2)])
                    states.append(State(pos_of_state, angle_of_state))
                    
                # Update angle and position
                pos = center + (rho + eps) * np.array([np.cos(angle + np.pi / 2 - length / rho), np.sin(angle + np.pi / 2 - length / rho)])
                angle = (angle - length / rho) % (2 * np.pi)

            case Dir.L:
                center = pos + rho * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

                n = int(np.ceil(length / delta))
                for k in range(1, n):
                    angle_of_state = (angle + (length / rho) * (k / n)) % (2 * np.pi)
                    pos_of_state = center + (rho + eps) * np.array([np.cos(angle_of_state - np.pi / 2), np.sin(angle_of_state - np.pi / 2)])
                    states.append(State(pos_of_state, angle_of_state))
                    
                # Update angle and position
                pos = center + rho * np.array([np.cos(angle - np.pi / 2 + length / rho), np.sin(angle - np.pi / 2 + length / rho)])
                angle = (angle + length / rho) % (2 * np.pi)

    return states + [q_f]

def call_dubins(q_i: State, q_f: State, rho: float) -> Tuple[Tuple[Dir, Dir, Dir], Tuple[float, float, float]]:
    """Calls the dubins library and converts the results into the used formats."""
    int_to_dir_string = {
        0: (Dir.L, Dir.S, Dir.L),
        1: (Dir.L, Dir.S, Dir.R),
        2: (Dir.R, Dir.S, Dir.L),
        3: (Dir.R, Dir.S, Dir.R),
        4: (Dir.R, Dir.L, Dir.R),
        5: (Dir.L, Dir.R, Dir.L),
    }

    path_type_dubins = dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).path_type()
    segment_lengths: Tuple[float, float, float] = [
        dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).segment_length(0),
        dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).segment_length(1),
        dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).segment_length(2)
    ]

    return (int_to_dir_string[path_type_dubins], segment_lengths)