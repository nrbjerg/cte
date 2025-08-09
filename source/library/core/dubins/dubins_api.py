"""This module provides an API for working with the dubins library, to make it easier to work with."""
import numpy as np
from typing import Iterable, Tuple, List
from classes.data_types import State, Dir, Position, Angle, Vector
from copy import deepcopy
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numba as nb
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

def sample_dubins_path(q_i: State, sub_segment_types: Iterable[Dir], sub_segment_lengths: Iterable[float], rho: float, delta: float, flip_angles: bool = False) -> Iterable[State]:
    """Samples the dubins path specified by the arguments, and flips the angles of the states if necessary."""
    if flip_angles:
        for (pos, angle) in _sample_dubins_path(q_i.to_tuple(), sub_segment_types, sub_segment_lengths, rho, delta):
            yield State(pos, angle).angle_complement()
    else:
        for (pos, angle) in _sample_dubins_path(q_i.to_tuple(), sub_segment_types, sub_segment_lengths, rho, delta):
            yield State(pos, angle)
        
def _sample_dubins_path(q_i: Tuple[float, float, Angle], sub_segment_types: Iterable[Dir], sub_segment_lengths: Iterable[float], rho: float, delta: float) -> Iterable[Vector]:
    """Samples the dubins path specified by the arguments, and flips the angles of the states if necessary."""
    pos = np.array(q_i[:2])
    angle = q_i[2]
    
    # Go over the segments
    for seg_type, seg_length in zip(sub_segment_types, sub_segment_lengths):
        if seg_length == 0:
            continue

        match seg_type:
            case Dir.S:
                # Iterate over n total 
                n = int(np.ceil(seg_length / delta)) + 1
                offset = (1 / n) * np.array([np.cos(angle), np.sin(angle)])
                for k in range(1, n):
                    yield pos + k * offset, angle

                pos += n * offset

            case Dir.R:
                center = pos + rho * np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])

                n = int(np.ceil(seg_length / delta)) + 1
                for k in range(1, n):
                    angle_of_state = (angle - (seg_length / rho) * (k / n)) % (2 * np.pi)
                    pos_of_state = center + rho * np.array([np.cos(angle_of_state + np.pi / 2), np.sin(angle_of_state + np.pi / 2)])
                    yield (pos_of_state, angle_of_state)
                    
                # Update angle and position
                pos = center + rho * np.array([np.cos(angle + np.pi / 2 - seg_length / rho), np.sin(angle + np.pi / 2 - seg_length / rho)])
                angle = (angle - seg_length / rho) % (2 * np.pi)

            case Dir.L:
                center = pos + rho * np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

                n = int(np.ceil(seg_length / delta))
                for k in range(1, n):
                    angle_of_state = (angle + (seg_length / rho) * (k / n)) % (2 * np.pi)
                    pos_of_state = center + rho * np.array([np.cos(angle_of_state - np.pi / 2), np.sin(angle_of_state - np.pi / 2)])
                    yield (pos_of_state, angle_of_state)
                    
                # Update angle and position
                pos = center + rho * np.array([np.cos(angle - np.pi / 2 + seg_length / rho), np.sin(angle - np.pi / 2 + seg_length / rho)])
                angle = (angle + seg_length / rho) % (2 * np.pi)

    yield (pos, angle)

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

def compute_minimum_distance_to_point_from_dubins_path_segment(p: Position, q: Tuple[float, float, Angle], seg_types: Iterable[Dir], seg_lengths: Iterable[float], rho: float) -> float:
    """Simply a wrapper."""
    return _compute_minimum_distance_to_point_from_dubins_path_segment(p, q.to_tuple(), seg_types, seg_lengths, rho)

def _compute_minimum_distance_to_point_from_dubins_path_segment(p: Position, q: Tuple[float, float, Angle], seg_types: Iterable[Dir], seg_lengths: Iterable[float], rho: float) -> float:
    """Computes the minimum distance from a point p to a dubins path starting at q, and being defined by the segment types and segment lengths."""
    pos = np.array([q[0], q[1]])
    angle = q[2]
    
    minimum_distance = np.inf
    for seg_idx in range(len(seg_types)):
        seg_type = seg_types[seg_idx]
        seg_length = seg_lengths[seg_idx]

        if seg_type == Dir.S:
            # Base case 
            if seg_length == 0:
                minimum_distance = min(minimum_distance, np.linalg.norm(pos - p))
                continue

            # Otherwise project onto the line segment, to find the closest point.
            offset = seg_length * np.array([np.cos(angle), np.sin(angle)])
            t = np.dot(p - pos, offset) / (seg_length ** 2) 

            # Compute minimum distance to closest point 
            minimum_distance = min(minimum_distance, np.linalg.norm(p - (pos + max(0.0, min(t, 1.0)) * offset)))

            # Update q to the end of the segment
            pos += offset

        # NOTE: while this is not the cleanest code, doing to many checks for the seg_type was to slow.
        elif seg_type == Dir.L:
            angle_offset = np.pi / 2
            offset = rho * np.array([np.cos(angle + angle_offset), np.sin(angle + angle_offset)])
            center = pos + offset
            radians = seg_length / rho

            # Compute the end position of the arc in the original coordinate system
            end_pos = center + rho * np.array([np.cos(angle + radians - angle_offset), np.sin(angle + radians - angle_offset)])

            # Convert coordinate system so that q = (0, 0, pi / 2)
            rotation_angle = np.pi / 2 - angle
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
            p_transformed = rotation_matrix.dot(p - pos)
            center_transformed = rotation_matrix.dot(center - pos)
            end_pos_transformed = rotation_matrix.dot(end_pos - pos)

            # Check if the transformed point is within the angle sweep of the arc.
            # NOTE: The if is simply mirroring across the x-axis so that we only have to consider the case where seg_type == Dir.L
            angle_between_x_axis_to_p_transformed = np.arctan2(p_transformed[1], p_transformed[0] + 1)
            angle_between_x_axis_to_p_transformed = angle_between_x_axis_to_p_transformed if angle_between_x_axis_to_p_transformed > 0 else angle_between_x_axis_to_p_transformed + 2 * np.pi

            # If not check the distances at the endpoints.
            if angle_between_x_axis_to_p_transformed < radians:
                d = np.linalg.norm(center_transformed - p_transformed)
                minimum_distance = min(minimum_distance, d - rho if d > rho else rho - d)
            else:
                minimum_distance = min((minimum_distance, np.linalg.norm(p_transformed), np.linalg.norm(p_transformed - end_pos_transformed)))

            # Update q to the end of the arc.
            pos = end_pos 
            angle = (angle + radians) % (2 * np.pi)

        else:
            angle_offset = -np.pi / 2
            offset = rho * np.array([np.cos(angle + angle_offset), np.sin(angle + angle_offset)])
            center = pos + offset
            radians = seg_length / rho

            # Compute the end position of the arc in the original coordinate system
            end_pos = center - rho * np.array([np.cos(angle - radians + angle_offset), np.sin(angle - radians + angle_offset)])

            # Convert coordinate system so that q = (0, 0, pi / 2)
            rotation_angle = np.pi / 2 - angle
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])
            p_transformed = rotation_matrix.dot(p - pos)
            center_transformed = rotation_matrix.dot(center - pos)
            end_pos_transformed = rotation_matrix.dot(end_pos - pos)

            # Check if the transformed point is within the angle sweep of the arc.
            # NOTE: The if is simply mirroring across the x-axis so that we only have to consider the case where seg_type == Dir.L
            angle_between_x_axis_to_p_transformed = np.arctan2(p_transformed[1], 1 - p_transformed[0])
            angle_between_x_axis_to_p_transformed = angle_between_x_axis_to_p_transformed if angle_between_x_axis_to_p_transformed > 0 else angle_between_x_axis_to_p_transformed + 2 * np.pi

            # If not check the distances at the endpoints.
            if angle_between_x_axis_to_p_transformed < radians:
                d = np.linalg.norm(center_transformed - p_transformed)
                minimum_distance = min(minimum_distance, d - rho if d > rho else rho - d)
            else:
                minimum_distance = min((minimum_distance, np.linalg.norm(p_transformed), np.linalg.norm(p_transformed - end_pos_transformed)))

            # Update q to the end of the arc.
            pos = end_pos 
            angle = (angle - radians) % (2 * np.pi)

    return minimum_distance 

if __name__ == "__main__":
    pos = np.array([1, 0])
    q = State(np.array([0, 1]), 0)
    seg_types = [Dir.R, Dir.S, Dir.L]
    seg_lengths = [1, 1, 1]
    rho = 1 
    
    # Plot the path
    plt.scatter(pos[0], pos[1], color = "tab:blue", zorder=3)
    plot_path(q, seg_types, seg_lengths, rho)

    # Example usage of compute_minimum_distance_to_point_from_dubins_path_segment
    print(compute_minimum_distance_to_point_from_dubins_path_segment(pos, q, seg_types, seg_lengths, rho))
    plt.show()