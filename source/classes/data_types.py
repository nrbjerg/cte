from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from matplotlib import pyplot as plt 
from matplotlib import patches
import random

Position = ArrayLike
Velocity = ArrayLike
Matrix = ArrayLike
Angle = float

@dataclass
class State:
    """Models a state in SE(2), that is R^2 x [0, 2pi)"""
    pos: Position
    angle: Angle

    def to_tuple(self) -> Tuple[float, float, float]:
        """Converts the state to a tuple."""
        return (self.pos[0], self.pos[1], self.angle)

    def angle_complement(self) -> State:
        """Returns the same state, with the angle fliped 180 degrees."""
        return State(self.pos, (np.pi + self.angle) % (2 * np.pi))

    def __repr__ (self) -> str:
        return f"{(round(float(self.pos[0]), 2), round(float(self.pos[1]), 2), round(self.angle, 2))}"

class AngleInterval:

    def __init__ (self, a: float, b: float):
        """Used to model an angle interval between the angles a and b."""
        self.a = a % (2 * np.pi)
        self.b = b % (2 * np.pi)

    def intersects(self, other: AngleInterval) -> bool:
        """Checks if the two angle intervals intercepts each other"""
        return any([self.contains(other.a), self.contains(other.b), 
                    other.contains(self.a), other.contains(self.b)])
        # FIXME I dont know if the last line is neccesary.

    def contains(self, psi: Angle) -> bool:
        """Checks if psi is contained within the interval"""
        if self.a < self.b:
            return self.a <= psi and psi <= self.b
        else:
            # Check if it is in the compliment of [b, a]
            return (self.b <= psi and psi < 2 * np.pi) or (psi >= 0 and psi <= self.a)
        
    def plot(self, ancour: Position, r: float, color: str, alpha: float):
        """Plots the angle as a circle arc of radius r with a center at the ancour point"""
        # The angles from the centers 
        a0, b0 = (self.a + np.pi) % (2 * np.pi), (self.b + np.pi) % (2 * np.pi)
        if a0 < b0:
            angles = np.linspace(a0, b0, 50) 
        else:
            angles = np.linspace(a0, (2 * np.pi) * (a0 // (2 * np.pi) + 1) + b0, 50)

        points_on_arc = np.vstack((ancour[0] + r * np.cos(angles),  
                                   ancour[1] + r * np.sin(angles)))
        points = np.vstack([points_on_arc.T, ancour])

        cone = patches.Polygon(points, closed=True, color = color, alpha = alpha)
        plt.gca().add_patch(cone)

    def generate_random_uniform_angle(self) -> Angle:
        """Generates a random angle from the angle interval, uniformly."""
        if self.a < self.b:
            return np.random.uniform(self.a, self.b)
        
        # Either generate an angle above or below the x axis depending on the proporition of the 
        # propotion of the arc described by the angle interval being above / below the x axis.
        elif random.choices([0, 1], weights = [2 * np.pi - self.a, self.b], k = 1)[0] == 1:
            return np.random.uniform(self.a, 2 * np.pi)
        else:
            return np.random.unifomr(0, self.b)

    def __repr__ (self) -> str:
        return f"[{round(self.a, 3)}; {round(self.b, 3)}]"
