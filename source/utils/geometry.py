from typing import Tuple
from classes.data_types import Position
import numpy as np


# REF: https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect/256123#256123 
def compute_intersection_points_of_cirlces(centers: Tuple[Position, Position], radii: Tuple[float, float]) -> Tuple[Position, Position]:
    """Computes the intersection points of two circles."""
    #x1, y1, r1, x2, y2, r2):
    difference = centers[0] - centers[1] 
    distance = np.linalg.norm(difference)

    if not (abs(radii[0] - radii[1]) <= distance and distance <= sum(radii)):
        """ No intersections """
        raise ValueError("Was given a pair of invalid cirlces.")

    """ intersection(s) should exist """
    midpoint = 1 / 2 * (centers[0] + centers[1])

    scalar_of_second_term = (radii[0]**2 + radii[1]**2) / (2 * distance**2) 
    second_term = scalar_of_second_term * (centers[1] - centers[0])

    scalar_of_third_term = 1 / 2 * np.sqrt(4 * scalar_of_second_term - ((radii[0]**2 - radii[1]**2)**2 / (distance**4)) - 1)
    third_term = scalar_of_third_term * np.array(centers[1][1] - centers[0][1], centers[1][0] - centers[0][0])

    return (midpoint + second_term + third_term, midpoint + second_term - third_term)
