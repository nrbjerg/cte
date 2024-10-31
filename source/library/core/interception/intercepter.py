from typing import List 
from classes.data_types import Position

class InterceptionRoute:
    """Models a general route of an intercepter, which is then inherited from."""
    delay: float # The delay between when the targets starts moving and the intercepter starts moving.
    velocity: float # Is assumed to remain constant throughout the route.
    
    # Initial and terminal positions (normally denotes the position of the source and sink nodes.)
    initial_position: Position
    terminal_position: Position

    # Contains a list of interception positions
    interception_points: List[Position]
    route_indicies: List[int] # The indicies of the routes which are intercepted.

    def plot(self, c: str = "tab:red", show: bool = True):
        """Plots the interception route."""
        pass # NOTE: should be implemented in the children

    def get_position(self, time: float) -> Position:
        """Computes the position of the euclidian interceptor at a given time."""
        pass # NOTE: should be implemented in the children

    def _distance_travelled_until_interception(self, i: int) -> float:
        """Computes the distance travelled until the i'th interception."""
        pass # NOTE: should be implemented in the children

    def time_until_interception(self, i: int) -> float:
        """Computes the time until the interception the i'th interception."""
        return self._distance_travelled_until_interception(i) / self.velocity + self.waiting_time
