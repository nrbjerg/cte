import dubins
from classes.data_types import State
from typing import List, Any
import matplotlib.pyplot as plt
import numpy as np 
from functools import cached_property

class DubinsRoute:
    """Models a dubins route with a given list of states."""
    states: List[State]
    rho: float
    paths: List[dubins._DubinsPath]

    def __init__ (self, states: List[State], rho: float):
        """Initializes the dubins route and computes the optimal paths between the states."""
        self.states = states
        self.rho = rho
        self.paths = [dubins.shortest_path(initial.to_tuple(), terminal.to_tuple(), rho) for initial, terminal in zip(states[:-1], states[1:])]

    @cached_property
    def length(self) -> float:
        """Computes the length of the dubins paths in the route."""
        return sum(path.path_length() for path in self.paths) 
    
    def plot(self, color: str = "black", plot_states: bool = False, show: bool = False):
        """Produces a plot fo a given dubins route, with a turning radius of rho."""
        plt.style.use("bmh")

        # Plot the states for illustation.
        if plot_states:
            for state in self.states:
                plt.scatter(state.pos[0], state.pos[1], c = color)
                xs = [state.pos[0], state.pos[0] + self.rho * np.cos(state.angle)] 
                ys = [state.pos[1], state.pos[1] + self.rho * np.sin(state.angle)]
                plt.plot(xs, ys, c = color) 

        for path in self.paths:
            configurations = np.array(path.sample_many(self.rho / 10)[0])
            plt.plot(configurations[:, 0], configurations[:, 1], c = color)
        
        if show:
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.show()
        
if __name__ == "__main__":
    route = DubinsRoute([State(np.array([0, 0]), 2 * np.pi / 3), State(np.array([1, 0.5]), 4 * np.pi / 3), State(np.array([1, -0.5]), -4 * np.pi / 3)], 0.2)
    print(route.length)
    route.plot(show = True)
