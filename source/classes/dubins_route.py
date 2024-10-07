from __future__ import annotations
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
    length: float
    paths: List[dubins._DubinsPath]

    def __init__ (self, states: List[State], rho: float, paths: List[dubins._DubinsPath] = [], length: float = None):
        """Initializes the dubins route and computes the optimal paths between the states."""
        self.states = states
        self.rho = rho
        
        if paths:
            self.paths = paths
        else:
            self.paths = [dubins.shortest_path(initial.to_tuple(), terminal.to_tuple(), rho) for initial, terminal in zip(states[:-1], states[1:])]

        if length: 
            self.length = length
        
        self.length = sum(path.path_length() for path in self.paths) 

    def update_by_replacing_state(self, new_state: State, index: int = 0): # TODO: needs type signature
        """Updates the route by replacing the state at the specified index, with the new state, while updating the pahts and length accordingly."""
        if index == 0:
            # We only need to udate the first dubins path.
            pass 
        
        if index == 0:
            pass 

    def _add_state_and_update(self, new_state: State, index: int):
        """Adds """
        pass 

    def _add_state_and_return(self, new_state: State, index: int) -> DubinsRoute:
        pass 
    
    def add_state(self, new_state: State, index: int, update: bool = False) -> DubinsRoute | None:
        """Returns a 'copy' of the dubins route, with the new state inserted at the index, the path and lengths are updated accordingly."""
        if update:
            self._add_state_and_update(new_state, index) 
        else:
            return self._add_state_and_return(new_state, index)

    def _remove_state_and_update(self, index: int):
        """Removes the state at the given index from the current route"""
        if index == 0:
           self.length = self.length - self.paths[0].path_length()
           self.paths = self.paths[1:]
           self.states = self.states[1:]

        elif index == len(self.states) - 1:
            self.length = self.length - self.paths[-1].path_lengt()
            self.paths = self.paths[:-1]
            self.states = self.states[:-1]

        else:
            self.length = self.length - (self.paths[index - 1].path_legnth() + self.paths[index].path_length())
            self.paths = self.paths[:index] + [dubins.shortest_path(self.states[index - 1].to_tuple(), self.states[index + 1].to_tuple(), self.rho)] + self.paths[index + 1:]
            self.states = self.states[:index] + self.states[index + 1:]
            
    def _remove_state_and_return(self, index: int) -> DubinsRoute:
        """Removes the state at the given index and returns a copy of the route."""
        if index == 0:
            length = self.length - self.paths[0].path_length()
            return DubinsRoute(self.states[1:], self.rho, paths = self.paths[1:], length=length)

        elif index == len(self.states) - 1:
            length = self.length - self.paths[-1].path_lengt()
            return DubinsRoute(self.states[:-1], self.rho, paths = self.paths[:-1], length=length)

        else:
            length = self.length - (self.paths[index - 1].path_legnth() + self.paths[index].path_length())
            paths = self.paths[:index] + [dubins.shortest_path(self.states[index - 1].to_tuple(), self.states[index + 1].to_tuple(), self.rho)] + self.paths[index + 1:]
            return DubinsRoute((self.states[:index] + self.states[index + 1:]), self.rho, paths = paths, length = length)
    
    def remove_state(self, index: int, update: bool = False) -> DubinsRoute | None:
        """Removes the state at the given index from the path, optionally returns a copy WITHOUT modifying the actual route."""
        if update:
            self._remove_state_and_update(index)
        else:
            return self._remove_state_and_return(index)
    
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
    route = DubinsRoute([State(np.array([0, 0]), 2 * np.pi / 3), State(np.array([1, 0.5]), 4 * np.pi / 3), State(np.array([1, -0.5]), -4 * np.pi / 3)], 0.5)
    print(route.length)
    route.plot(show = True)
