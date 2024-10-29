# %%
from classes.route import Route
from classes.data_types import Position
from scipy.optimize import bisect
from typing import List
import numpy as np 
from classes.node import Node
from matplotlib import pyplot as plt

#def compute_euclid_interception(pos: Position, vel: float, route: Route, lock_on_time: float = 0, eps: float = 0.001) -> Position | None:
#    """Computes the position of the closest euclidian intercept on the route."""
#    if vel < 1:
#        raise ValueError("Expected the pursuers velocity to be greater than 1, which is the velocity of the targets.")
#
#    # NOTE: the three points should probably not be colinear, if we want to use this solution method?
#    relative_times = [time - lock_on_time for time in route.visit_times]
#    print(f"Relative times to reach nodes: {relative_times}")
#    start_index = min([idx for idx in range(len(relative_times)) if relative_times[idx] >= 0])
#    for i, (fst, snd) in enumerate(zip(route.nodes[start_index:-1], route.nodes[start_index + 1:]), start_index): 
#        # First find the correct line segment.
#        #if np.linalg.norm(fst.pos - pos) / vel <= relative_times[i] or np.linalg.norm(snd.pos - pos) / vel > relative_times[i + 1]:
#            #continue
#
#        print(f"\nEndpoints and times for intercept: <{fst.pos}: {np.linalg.norm(fst.pos - pos) / vel}, {relative_times[i]}>, <{snd.pos}: {np.linalg.norm(snd.pos - pos) / vel}, {relative_times[i + 1]}>")
#
#        # Parametic equation of the line segment becomes f(t) = v t + w with t \in [0; 1]
#        v = snd.pos - fst.pos
#        w = fst.pos - pos
#
#        # Compute the time of interception (under the assumption that v > 1)
#        temp0 = relative_times[i + 1] - relative_times[i]
#        temp1 = relative_times[i + 1] ** 2 + relative_times[i] ** 2 - 2 * relative_times[i + 1] * relative_times[i]
#
#        a = np.dot(v, v) / temp1 
#        b = 2 * relative_times[i] * np.dot(v, v) / temp1 + 2 * np.dot(v, w) / temp0 
#        c = np.dot(v, v) * relative_times[i] ** 2 / temp1 + 2 * relative_times[i] * np.dot(v, w) / temp0 + np.dot(w, w)
#
#        solutions = np.abs(np.roots([a - (vel ** 2), b, c]))
#        #print(f"Evaluations: {[(a / (vel ** 2) - 1) * (t ** 2) + b / (vel ** 2) * t + c / (vel ** 2) for t in solutions]}")
#        print(f"Solutions to polynomial equation: {solutions}")
#
#        for solution in solutions:
#            # Check if the solution yields a point on the line segment between fst and snd.
#            point = route.get_position_at_time(solution + lock_on_time)
#            time_to_reach_point = np.linalg.norm(point - pos) / vel
#            print(f" - Line segment: {fst.pos}, {snd.pos}:\n   Interception point: {point} \n   Times to reach interception point: {time_to_reach_point, solution + lock_on_time}")
#            if (solution > relative_times[i] and solution <= relative_times[i + 1]) and (time_to_reach_point >= solution - eps and time_to_reach_point <= solution + eps):
#                return point
#    
#    print("Warning: Did not find an interception point!")
#    return None


# NOTE: This version computes the interception points using newton-raphson.
def compute_euclid_interception(pos: Position, vel: float, route: Route, lock_on_time: float = 0, eps: float = 0.001, plot: bool = False) -> Position | None:
    """Computes the position of the closest euclidian intercept on the route."""
    if vel < 1:
        raise ValueError("Expected the pursuers velocity to be greater than 1, which is the velocity of the targets.")
    
    f = lambda t: np.linalg.norm(route.get_position_at_time(t + lock_on_time) - pos) - t * vel
    relative_times = [time - lock_on_time for time in route.visit_times]
    #print(f"Relative times to reach nodes: {relative_times}")
    try:
        start_index = min([idx for idx in range(len(relative_times)) if relative_times[idx] >= 0])
    except ValueError:
        return route.nodes[-1].pos # NOTE: in this case the UAV has already reached the sink

    if plot:
        ts = np.arange(0, relative_times[-1], 0.05) 
        plt.plot(ts, [0 for _ in ts])
        plt.plot(ts, [f(t) for t in ts])
        plt.show()


    # Consider the case where we dont reach the next node before the interception.
    if np.sign(f(0)) != np.sign(f(relative_times[start_index])):
        root = bisect(f, 0, relative_times[start_index])
        print(f"Root: {root}")

        # Check if the solution yields a point on the line segment between fst and snd.
        point = route.get_position_at_time(root + lock_on_time)
        time_to_reach_point = np.linalg.norm(point - pos) / vel
        #print(f" - Line segment: {fst.pos}, {snd.pos}:\n   Interception point: {point} \n   Times to reach interception point: {time_to_reach_point, root + lock_on_time}")
        if (root > 0 and root <= relative_times[start_index]) and (time_to_reach_point >= root - eps and time_to_reach_point <= root + eps):
            return point
    
    # Consider the other cases.
    for i in range(start_index, len(relative_times)): 
        # First find the correct line segment.
        if np.sign(f(relative_times[i])) == np.sign(f(relative_times[i + 1])):
            continue

        #print(f"\nEndpoints and times for intercept: <{fst.pos}: {np.linalg.norm(fst.pos - pos) / vel}, {relative_times[i]}>, <{snd.pos}: {np.linalg.norm(snd.pos - pos) / vel}, {relative_times[i + 1]}>")

        root = bisect(f, relative_times[i], relative_times[i + 1])

        # Check if the solution yields a point on the line segment between fst and snd.
        point = route.get_position_at_time(root + lock_on_time)
        time_to_reach_point = np.linalg.norm(point - pos) / vel
        #print(f" - Line segment: {fst.pos}, {snd.pos}:\n   Interception point: {point} \n   Times to reach interception point: {time_to_reach_point, root + lock_on_time}")
        if (root > relative_times[i] and root <= relative_times[i + 1]) and (time_to_reach_point >= root - eps and time_to_reach_point <= root + eps):
            return point
    
    print("Warning: Did not find an interception point!")
    return None


class EuclidianInterceptionRoute:
    """Models a route of an euclidian intercepter."""
    waiting_time: float
    velocity: float
    initial_position: Position
    interception_points: List[Position]
    terminal_position: Position 

    def __init__(self, initial_position: Position, ks: List[int], routes: List[Route], vel: float, terminal_position: Position, waiting_time: float = 0):
        """Initializes the euclidian interception route, and computes its path using the compute_euclid_interception function."""
        self.waiting_time = waiting_time
        self.velocity = vel
        self.initial_position = initial_position
        time = waiting_time
        self.interception_points = []
        self.terminal_position = terminal_position

        for k in ks:
            # Compute the interception point of each k, moving from the latest position.
            if len(self.interception_points) != 0:
                self.interception_points.append(compute_euclid_interception(self.interception_points[-1], vel, routes[k], lock_on_time=time))
            else:
                self.interception_points.append(compute_euclid_interception(initial_position, vel, routes[k], lock_on_time=time))

            # Simply compute the time spent to get from the previous position to the new position.
            if len(self.interception_points) >= 2:
                time += np.linalg.norm(self.interception_points[-1] - self.interception_points[-2]) / vel
            else:
                time += np.linalg.norm(self.interception_points[-1] - self.initial_position) / vel


    def plot(self, c: str = "tab:red", show: bool = True):
        """Plots the euclidian interception route using matplotlib"""
        xs, ys = [self.initial_position[0]], [self.initial_position[1]]
        for point in self.interception_points:
            xs.append(point[0])
            ys.append(point[1])

        xs.append(self.terminal_position[0])
        ys.append(self.terminal_position[1])

        plt.plot(xs, ys, c=c)
        plt.scatter(xs[:-1], ys[:-1], marker="D", c=c)
        if show:
            plt.show()

    def get_position_at_time(self, t: float) -> Position:
        """Computes the position of the euclidian interceptor at time t."""
        if t < self.waiting_time:
            return self.initial_position
        
        remaining_time = t - self.waiting_time
        for idx, interception_point in enumerate(self.interception_points):
            # Compute the time to get to the interception point
            if idx == 0:
                time_to_reach_interception_point = np.linalg.norm(self.initial_position - interception_point) / self.velocity
            else:
                time_to_reach_interception_point = np.linalg.norm(self.interception_points[idx - 1] - interception_point) / self.velocity
            
            # If we dont reach the next interception point compute the position on the line between 
            # the last interception point and the next interception point.
            if time_to_reach_interception_point > remaining_time:
                proportion_of_way_there = remaining_time / time_to_reach_interception_point
                if idx != 0:
                    return self.interception_points[idx - 1] * (1 - proportion_of_way_there) + interception_point * proportion_of_way_there
                else:
                    return self.initial_position * (1 - proportion_of_way_there) + interception_point * proportion_of_way_there

            else:
                remaining_time -= time_to_reach_interception_point

        
        time_to_reach_terminal_position = np.linalg.norm(self.terminal_position - self.interception_points[-1]) / self.velocity
        proportion_of_way_there = remaining_time / time_to_reach_terminal_position
        if proportion_of_way_there > 1:
            return self.interception_points[-1]
        else:
            return self.interception_points[-1] * (1 - proportion_of_way_there) + self.terminal_position * proportion_of_way_there


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
