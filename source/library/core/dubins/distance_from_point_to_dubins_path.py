import numpy as np
import numba as nb
from library.core.dubins import dubins_api, relaxed_dubins
from classes.data_types import State, Position
import matplotlib.pyplot as plt 

def compute_length_from_arc(pos: Position, q: State, rho: float, radians_traversed_on_arc: float, dir: relaxed_dubins.Direction) -> float:
    """Computes the minimum distance from the 'pos' to the arc, starting at q."""
    if dir == relaxed_dubins.Direction.RIGHT:
        pass 

    else: # TODO 
        return np.inf

def compute_length_from_line_segment(pos: Position, q: State, seg_length: float) -> float:
    """Computes the minimum distance from the 'pos' to the line segment, starting at q."""
    direction = np.array([np.cos(q.angle), np.sin(q.angle)])
    t = max(0, min(seg_length, np.dot((pos - q.pos), direction)))
    return np.linalg.norm(q.pos + direction * t - pos) 

def compute_minimum_distance_from_dubins_path(pos: Position, q0: State, q1: State, rho: float) -> float:
    """Computes the minimum distance between p and the dubins path between q0 and q1 with minimum turning radius rho."""
    seg_types, seg_lengths = dubins_api.call_dubins(q0, q1, rho)
    min_distance = np.inf
    q = q0
    for seg_type, seg_length in zip(seg_types, seg_lengths):
        if seg_type == relaxed_dubins.Direction.SAIGHT:
            min_distance = min(min_distance, compute_length_from_line_segment(pos, q, seg_length))
            q.pos += np.array([np.cos(q.angle), np.sin(q.angle)]) * seg_length

        else:
            rotation_angle = np.pi / 2 - q.angle
            rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], 
                                        [np.sin(rotation_angle), np.cos(rotation_angle)]])

            radians_traversed_on_arc = seg_length / rho
            min_distance = min(min_distance, compute_length_from_arc(pos, q, rho, radians_traversed_on_arc, seg_type))

            if seg_type == relaxed_dubins.Direction.RIGHT:
                theta = np.pi - radians_traversed_on_arc
                q.pos = rotation_matrix.T.dot(np.array([rho, 0])) + q.pos + rotation_matrix.T.dot(rho * np.array([np.cos(theta), np.sin(theta)]))
                q.angle -= radians_traversed_on_arc

            else:
                theta = radians_traversed_on_arc
                q.pos = rotation_matrix.T.dot(np.array([-rho, 0])) + q.pos + rotation_matrix.T.dot(rho * np.array([np.cos(theta), np.sin(theta)]))
                q.angle += radians_traversed_on_arc

    return min_distance

def compute_minimum_distance_from_relaxed_dubins_path(pos: Position, q: State, p: Position, rho: float) -> float:
    """Computes the minimum distance between pos and the relaxed dubins path from q to p, with minimum turning radius rho."""
    relaxed_dubins_path = relaxed_dubins.compute_relaxed_dubins_path(q, p, rho, plot = True)
    
    # 1. Compute length from initial arc.
    if type(relaxed_dubins_path) == relaxed_dubins.CSPath:
        radians_traversed_on_arc = relaxed_dubins_path.radians_traversed_on_arc
        direction = relaxed_dubins_path.direction 
    else:
        radians_traversed_on_arc = relaxed_dubins_path.radians_traversed_on_arcs[0]
        direction = relaxed_dubins_path.directions[0]
    
    min_length = compute_length_from_arc(pos, q, rho, radians_traversed_on_arc, direction)

    # 2. Update q
    rotation_angle = np.pi / 2 - q.angle
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], 
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    if direction == relaxed_dubins.Direction.RIGHT:
        theta = np.pi - radians_traversed_on_arc
        q.pos = rotation_matrix.T.dot(np.array([rho, 0])) + q.pos + rotation_matrix.T.dot(rho * np.array([np.cos(theta), np.sin(theta)]))
        q.angle -= radians_traversed_on_arc

    else:
        theta = radians_traversed_on_arc
        q.pos = rotation_matrix.T.dot(np.array([-rho, 0])) + q.pos + rotation_matrix.T.dot(rho * np.array([np.cos(theta), np.sin(theta)]))
        q.angle += radians_traversed_on_arc
    
    plt.scatter(q.pos[0], q.pos[1]) 
    print(q.angle)
    # 3. Compute length from final segment 
    if type(relaxed_dubins_path) == relaxed_dubins.CSPath:
        seg_legnth = relaxed_dubins_path.length - relaxed_dubins_path.radians_traversed_on_arc * rho
        min_length = min(min_length, compute_length_from_line_segment(pos, q, seg_legnth))
    
    if type(relaxed_dubins_path) == relaxed_dubins.CCPath:
        min_length = min(min_length, compute_length_from_arc(pos, q, rho, relaxed_dubins_path.radians_traversed_on_arcs[1], relaxed_dubins_path.directions[1]))
                                    
    return min_length

if __name__ == "__main__":
    plt.style.use("ggplot")
    pos = np.array([2, -1])
    q = State(np.array([0, 0]), np.pi / 2)
    p = np.array([-0.5, 0.4])
    plt.scatter(pos[0], pos[1])
    print(compute_minimum_distance_from_relaxed_dubins_path(pos, q, p, rho = 1))
    plt.show()

    
