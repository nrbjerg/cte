# %%
"""
Implementation of dubins shortest path, using the math from: https://bpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/e/700/files/2014/04/Dubins_Set_Robotics_2001.pdf and inspiration from: https://github.com/AndrewWalker/pydubins/blob/master/dubins/src/dubins.c
"""
from classes.data_types import State, Angle
from copy import deepcopy
from enum import IntEnum
from typing import Tuple, Optional, Iterable
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import dubins

def mod2pi(theta: float) -> Angle:
    return (theta - (2 * np.pi) * np.floor(theta / (2 * np.pi)))

class Dir(IntEnum):
    """Models a direction of a turn"""
    R = 0
    L = 1
    S = 2

    def __repr__ (self) -> str:
        if self == Dir.R: return "R"
        if self == Dir.L: return "L"
        else: return "S"

def LSL_path(d: float, alpha: Angle, beta: Angle) -> Optional[Tuple[float, float, float]]:
    """Computes the lengths of the LSL path if it is posible to traverse such a path."""
    if (p_sq := (2 + np.square(d) - 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) - np.sin(beta)))) >= 0:
        tmp1 = np.atan2(np.cos(beta) - np.cos(alpha), d + np.sin(alpha) - np.sin(beta))
        return (mod2pi(tmp1 + mod2pi(- alpha)), np.sqrt(p_sq), mod2pi(beta - tmp1))

def RSR_path(d: float, alpha: Angle, beta: Angle) -> Optional[Tuple[float, float, float]]:
    """Computes the lengths of the RSR path if it is posible to traverse such a path."""
    if (p_sq := 2 + np.square(d) - (2 * np.cos(alpha - beta)) + 2 * d * (np.sin(beta) - np.sin(alpha))) >= 0:
        tmp1 = np.atan2(np.cos(alpha) - np.cos(beta), (d - np.sin(alpha) + np.sin(beta)))
        return (mod2pi(alpha - tmp1), np.sqrt(p_sq), mod2pi(tmp1 - beta))

def LSR_path(d: float, alpha: Angle, beta: Angle) -> Optional[Tuple[float, float, float]]:
    """Computes the lengths of the LSR path if it is posible to traverse such a path."""
    if (p_sq := np.square(d) - 2 + (2 * np.cos(alpha - beta)) + 2 * d * (np.sin(alpha) + np.sin(beta))) >= 0:
        p = np.sqrt(p_sq)
        tmp0 = np.atan2(- np.cos(alpha) - np.cos(beta), (d + np.sin(alpha) + np.sin(beta))) - np.atan2(-2, p)
        return (mod2pi(tmp0 - alpha), p, mod2pi(tmp0 - beta))
    
def RSL_path(d: float, alpha: Angle, beta: Angle) -> Optional[Tuple[float, float, float]]:
    """Computes the lengths of the RSL path if it is posible to traverse such a path."""
    if (p_sq := np.square(d) - 2 + (2 * np.cos(alpha - beta)) - 2 * d * (np.sin(alpha) + np.sin(beta))) >= 0:
        p = np.sqrt(p_sq)
        tmp0 = np.atan2(np.cos(alpha) + np.cos(beta), (d - np.sin(alpha) - np.sin(beta))) - np.atan2(2, p)
        return (mod2pi(alpha - tmp0), p, mod2pi(beta - tmp0))

def RLR_path(d: float, alpha: Angle, beta: Angle) -> Optional[Tuple[float, float, float]]:
    """Computes the lengths of the RLR path if it is posible to traverse such a path."""
    tmp0 = (6 - np.square(d) + 2 * np.cos(alpha - beta) + 2 * d * (np.sin(alpha) - np.sin(beta))) / 8
    if abs(tmp0) <= 1:
        phi = np.atan2(np.cos(alpha) - np.cos(beta), d  - np.sin(alpha) + np.sin(beta))
        p = mod2pi((2 * np.pi) - np.arccos(tmp0))
        t = mod2pi(alpha - phi + mod2pi(p / 2))
        return (t, p, mod2pi(alpha - beta - t + mod2pi(p)))

def LRL_path(d: float, alpha: Angle, beta: Angle) -> Optional[Tuple[float, float, float]]:
    """Computes the lengths of the LRL path if it is posible to traverse such a path."""
    tmp0 = (6 - np.square(d) + 2 * np.cos(alpha - beta) + 2 * d * (np.sin(beta) - np.sin(alpha))) / 8
    if abs(tmp0) <= 1:
        phi = np.atan2(np.cos(alpha) - np.cos(beta), d  + np.sin(alpha) - np.sin(beta))
        p = mod2pi(2 * np.pi - np.arccos(tmp0))
        t = mod2pi(-alpha - phi + p / 2)
        return (t, p, mod2pi(beta - alpha - t + mod2pi(p)))

def dubins_shortest_path(q_i: State, q_f: State, rho: float):
    """Computes the shortest dubins path from q_i to q_f with a minimum turning radius of rho"""
    # 1. Convert to cordinate system where q_i = (0, 0, alpha) and q_f = (d, 0, beta)
    dx, dy = q_f.pos[0] - q_i.pos[0], q_f.pos[1] - q_i.pos[1]
    D = np.linalg.norm((dx, dy))
    print(D)
    d = D / rho

    theta = mod2pi(np.atan2(dy, dx)) if d != 0 else 0
    alpha = mod2pi(q_i.angle - theta)
    beta = mod2pi(q_f.angle - theta)

    # 2. Find shortest path.
    shortest_length = np.inf
    shortest_params = None
    shortest_path_type = None
    for (func, path_type) in zip([LSL_path, RSR_path, LSR_path, RSL_path, RLR_path, LRL_path], 
                                 [(Dir.L, Dir.S, Dir.L), (Dir.R, Dir.S, Dir.R), (Dir.L, Dir.S, Dir.R),
                                  (Dir.R, Dir.S, Dir.L), (Dir.R, Dir.L, Dir.R), (Dir.L, Dir.R, Dir.L)]):
        params = func(d, alpha, beta)
        if params != None:
            print(path_type, params)
        if (params != None) and ((length := sum(params)) < shortest_length):
            shortest_length = length
            shortest_params = params
            shortest_path_type = path_type

    return ([p * rho for p in shortest_params], shortest_path_type)

def plot_path(q: State, segment_types: Iterable[Dir], segment_lengths: Iterable[float], rho: float, color: str = "tab:green"):
    """Plots an arbitary dubins path, of the given segment types and parameters"""
    pos = deepcopy(q.pos)
    angle = deepcopy(q.angle)
    for i, (typ, length) in enumerate(zip(segment_types, segment_lengths)):
        match typ:
            case Dir.S:
                plt.plot([pos[0], pos[0] + np.cos(angle) * length],
                         [pos[1], pos[1] + np.sin(angle) * length], color = color, zorder=2)
                pos[0] += np.cos(angle) * length
                pos[1] += np.sin(angle) * length
            case Dir.R:
                center = pos + rho * np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])
                plt.scatter(*center, color = "tab:red" if i == 0 else "tab:purple")
                arc = mpl.patches.Arc(center, 2 * rho, 2 * rho, theta2 = np.rad2deg(angle + np.pi / 2), theta1 = np.rad2deg(angle + np.pi / 2 - length / rho), edgecolor = color, zorder = 4)
                plt.gca().add_patch(arc) 
                pos = center + rho * np.array([np.cos(angle + np.pi / 2 - length / rho), np.sin(angle + np.pi / 2 - length / rho)])
                angle = mod2pi(angle - length / rho)

            case Dir.L:
                center = pos + rho * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
                plt.scatter(*center, color = "tab:red" if i == 0 else "tab:purple")
                arc = mpl.patches.Arc(center, 2 * rho, 2 * rho, theta1 = np.rad2deg(angle - np.pi / 2), theta2 = np.rad2deg(angle - np.pi / 2 + length / rho), edgecolor = color, zorder = 2)
                plt.gca().add_patch(arc) 
                pos = center + rho * np.array([np.cos(angle - np.pi / 2 + length / rho), np.sin(angle - np.pi / 2 + length / rho)])
                angle = mod2pi(angle + length / rho)

if __name__ == "__main__":
    rho = 1

    np.random.seed(0)
    for idx in range(100):
        q_i = State(np.random.uniform(-2 * rho, 2 * rho, size = 2), np.random.uniform(0, 2 * np.pi))
        q_f = State(np.random.uniform(-2 * rho, 2 * rho, size = 2), np.random.uniform(0, 2 * np.pi))
    
        results = dubins_shortest_path(q_i, q_f, rho)
        print(f"Results: {results}")
        legnth_from_dubins = dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).path_length()
        path_type_dubins = dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).path_type()
        path_type = results[1]
        segment_lengths = [
            dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).segment_length(0),
            dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).segment_length(1),
            dubins.shortest_path(q_i.to_tuple(), q_f.to_tuple(), rho).segment_length(2)
        ]

        int_to_dir_string = {
            0: (Dir.L, Dir.S, Dir.L),
            1: (Dir.L, Dir.S, Dir.R),
            2: (Dir.R, Dir.S, Dir.L),
            3: (Dir.R, Dir.S, Dir.R),
            4: (Dir.R, Dir.L, Dir.R),
            5: (Dir.L, Dir.R, Dir.L),
        }
        print(f"Segments: {segment_lengths}, type: {int_to_dir_string[path_type_dubins]}")
        length = sum(results[0])
        #configurations = np.array(dubins.path_sample(q_i.to_tuple(), q_f.to_tuple(), rho, 0.01)[0])
        #plt.plot(configurations[:, 0], configurations[:, 1], c = "tab:orange", zorder=2)
        if abs(length - legnth_from_dubins) > 0.001:
            q_i.plot()
            q_f.plot()
            plot_path(q_i, int_to_dir_string[path_type_dubins], segment_lengths, rho, color = "tab:blue")
            plot_path(q_i, results[1], results[0], rho, color = "tab:green")
            plt.show()


# %%
