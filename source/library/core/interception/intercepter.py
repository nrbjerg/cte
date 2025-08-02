from typing import List 
from classes.data_types import Position
from classes.route import Route

class InterceptionRoute:
    """Models a general route of an intercepter, which is then inherited from."""
    delay: float # The delay between when the targets starts moving and the intercepter starts moving.
    velocity: float # Is assumed to remain constant throughout the route.
    
    # Initial and terminal positions (normally denotes the position of the source and sink nodes.)
    initial_position: Position
    terminal_position: Position

    # Contains a list of interception positions
    interception_points: List[Position]
    interception_times: List[float]
    route_indices: List[int] # The indices of the routes which are intercepted.

    def __init__(self, route_indices: List[int], routes: List[Route], velocity: float, delay: float, initial_position: Position, terminal_position: Position):
        """Initializes the intercepter and performs initial computation of interception points and times."""
        pass # NOTE: should be implemented in the children

    def plot(self, c: str = "tab:purple", show: bool = True):
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
        return self._distance_travelled_until_interception(i) / self.velocity + self.delay

    def can_route_be_intercepted(self, route: Route, route_index: int) -> bool:
        """Checks if the given route can be intercepted after the final interception point."""
        pass # NOTE: should be implemented in the children
        
        if route_index == self.route_indices[-1]:
            return False
        


    