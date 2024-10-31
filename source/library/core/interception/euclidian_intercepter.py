# %%
from classes.route import Route
from classes.data_types import Position
from scipy.optimize import bisect
from typing import List
import numpy as np 
from classes.node import Node
from matplotlib import pyplot as plt
from library.core.interception.intercepter import InterceptionRoute

# NOTE: This version computes the interception points using newton-raphson.
def compute_euclid_interception(pos: Position, vel: float, route: Route, default_position: Position, lock_on_time: float = 0, eps: float = 0.001, plot: bool = False) -> Position | None:
    """Computes the position of the closest euclidian intercept on the route."""
    if vel < 1:
        raise ValueError("Expected the pursuers velocity to be greater than 1, which is the velocity of the targets.")
    
    f = lambda t: np.linalg.norm(route.get_position(t + lock_on_time) - pos) - t * vel

    relative_times = [time - lock_on_time for time in route.visit_times]
    try:
        start_index = min([idx for idx in range(len(relative_times)) if relative_times[idx] >= 0])
    except ValueError:
        return route.nodes[-1].pos # NOTE: in this case the UAV has already reached the sink

    if plot: # TODO: Remove?
        ts = np.arange(0, relative_times[-1], 0.05) 
        plt.plot(ts, [0 for _ in ts])
        plt.plot(ts, [f(t) for t in ts])
        plt.show()

    # Consider the case where we dont reach the next node before the interception.
    if np.sign(f(0)) != np.sign(f(relative_times[start_index])):
        root = bisect(f, 0, relative_times[start_index])

        # Check if the solution yields a point on the line segment between fst and snd.
        point = route.get_position(root + lock_on_time)
        time_to_reach_point = np.linalg.norm(point - pos) / vel
        if (root > 0 and root <= relative_times[start_index]) and (time_to_reach_point >= root - eps and time_to_reach_point <= root + eps):
            return point
    
    # Consider the other cases.
    for i in range(start_index, len(relative_times) - 1): 
        # First find the correct line segment.
        if np.sign(f(relative_times[i])) == np.sign(f(relative_times[i + 1])):
            continue

        root = bisect(f, relative_times[i], relative_times[i + 1])

        # Check if the solution yields a point on the line segment between fst and snd.
        point = route.get_position(root + lock_on_time)
        time_to_reach_point = np.linalg.norm(point - pos) / vel
        if (root > relative_times[i] and root <= relative_times[i + 1]) and (time_to_reach_point >= root - eps and time_to_reach_point <= root + eps):
            return point
    
    print("Warning: Did not find an interception point!")
    return default_position


class EuclidianInterceptionRoute(InterceptionRoute):
    """Models a route of an euclidian intercepter, that is an intercepter moving in a straight line to intercept the targets."""

    def __init__(self, initial_position: Position, route_indicies: List[int], routes: List[Route], vel: float, terminal_position: Position, delay: float = 0):
        """Initializes the euclidian interception route, and computes its path using the compute_euclid_interception function."""
        self.route_indicies = route_indicies
        self.delay = delay 
        self.velocity = vel
        self.initial_position = initial_position
        self.terminal_position = terminal_position

        # Compute first interception point
        self.interception_points = [compute_euclid_interception(initial_position, vel, routes[route_indicies[0]], self.terminal_position, lock_on_time=delay)]
        time = delay + np.linalg.norm(self.interception_points[-1] - self.initial_position) / vel

        for k in route_indicies[1:]:
            # Compute the interception point of each k, moving from the latest interception point.
            self.interception_points.append(compute_euclid_interception(self.interception_points[-1], vel, routes[k], self.terminal_position, lock_on_time=time))
            time += np.linalg.norm(self.interception_points[-1] - self.interception_points[-2]) / vel

    def plot(self, c: str = "tab:purple", show: bool = True, zorder: int = 5):
        """Plots the euclidian interception route using matplotlib"""
        xs = [self.initial_position[0]] + [point[0] for point in self.interception_points] + [self.terminal_position[0]]
        ys = [self.initial_position[1]] + [point[1] for point in self.interception_points] + [self.terminal_position[1]]

        plt.plot(xs, ys, c=c, zorder=zorder)
        plt.scatter(xs[1:-1], ys[1:-1], marker="D", c=c, zorder=zorder) # NOTE: Skip the first and final points (that is the source and sink!)
        if show:
            plt.show()

    def get_position(self, time: float) -> Position:
        """Computes the position of the euclidian interceptor at time t."""
        if time < self.delay:
            return self.initial_position
        
        remaining_time = time - self.delay

        # Check if we reaches the first interception point, before time is up.
        time_to_reach_fst = np.linalg.norm(self.initial_position - self.interception_points[0]) / self.velocity
        if time_to_reach_fst > remaining_time:
            proportion = remaining_time / time_to_reach_fst
            return self.initial_position * (1 - proportion) + self.interception_points[0] * proportion

        else:
            remaining_time -= time_to_reach_fst

        # For each pair of interception points, check that we make it to the second interception point before time is up.
        for fst, snd in zip(self.interception_points[:-1], self.interception_points[1:]):
            time_to_reach_snd = np.linalg.norm(fst - snd) / self.velocity
            
            if time_to_reach_snd > remaining_time:
                proportion = remaining_time / time_to_reach_snd
                return fst * (1 - proportion) + snd * proportion

            else:
                remaining_time -= time_to_reach_snd
        
        # Check if we reaches the final position before the time is up.
        time_to_reach_terminal_position = np.linalg.norm(self.terminal_position - self.interception_points[-1]) / self.velocity
        if time_to_reach_terminal_position < remaining_time:
            return self.terminal_position
        else:
            proportion = remaining_time / time_to_reach_terminal_position
            return self.interception_points[-1] * (1 - proportion) + self.terminal_position * proportion

    def _distance_travelled_until_interception(self, i: int) -> float:
        """Computes the distance travelled until the i'th interception."""
        total = np.linalg.norm(self.initial_position - self.interception_points[0]) 
        total += sum(np.linalg.norm(self.interception_points[j - 1] - self.interception_points[j]) for j in range(1, i + 1))
        return total


if __name__ == "__main__":
    v0 = Node(0, [], np.array((0, 2)), 0, 0, 0)
    v1 = Node(0, [], np.array((2, 2)), 0, 0, 0)
    v2 = Node(0, [], np.array((4, -2)), 0, 0, 0)
    vel = 1.6
    initial_pos = np.array((-4, -3))
    route = Route([v0, v1, v2])
    point = compute_euclid_interception(initial_pos, vel, route) 
    if point is not None:
        print(f"Interception happends at point: {point} with an interception time of {np.linalg.norm(point - initial_pos) / vel}")
