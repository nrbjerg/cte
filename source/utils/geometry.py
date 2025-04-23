from typing import Tuple
from classes.data_types import Position
import numpy as np
import numba as nb

# Shamelessly grabed from https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles 
@nb.njit()
def get_intersection_points_between_circles(c1: Position, r1: float, c2: Position, r2: float) -> Tuple[Position, Position]:
    """Computes the intersection points of two circles with centers at c1 and c2 and radiis of r1 and r2 respectively."""
    d = np.linalg.norm(c1 - c2)
    if d > r1 + r2 or d < np.abs(r1 - r2) or d == 0:
        raise ValueError(f"The circles are either non-intersecting or a fully intersecting, {(c1, r1)}, {(c2, r2)}")

    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h=np.sqrt(r1 ** 2 - a ** 2)
    
    x0 = c1[0] + a * (c2[0] - c1[0]) / d 
    y0 = c1[1] + a * (c2[1] - c1[1]) / d

    x1 = x0 + h * (c2[1] - c1[1]) / d
    y1 = y0 - h * (c2[0] - c1[0]) / d

    x2 = x0 - h * (c2[1] - c1[1]) / d
    y2 = y0 + h * (c2[0] - c1[0]) / d

    return (np.array([x1, y1]), np.array([x2, y2])) 

