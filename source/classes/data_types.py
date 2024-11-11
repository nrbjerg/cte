from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from matplotlib import pyplot as plt 
from matplotlib import patches

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
        if self.a < self.b:
            angles = np.linspace(self.a, self.b, 50) 
        else:
            angles = np.linspace(self.a, (2 * np.pi) * (self.a // (2 * np.pi) + 1) + self.b, 50)

        points_on_arc = np.vstack((ancour[0] + r * np.cos(angles),  
                                   ancour[1] + r * np.sin(angles)))
        points = np.vstack([points_on_arc.T, ancour])

        cone = patches.Polygon(points, closed=True, color = color, alpha = alpha)
        plt.gca().add_patch(cone)

        #plt.plot([ancour[0], ancour[0] + r * np.cos(self.theta)], [ancour[1], ancour[1] + r * np.sin(self.theta)])

    def __repr__ (self) -> str:
        return f"[{round(self.a, 3)}; {round(self.b, 3)}]"

if __name__ == "__main__":
    intervals = [AngleInterval(5, 1), AngleInterval(0, 1), AngleInterval(2, 4)]
    print(intervals[0].intersects(intervals[1]))
    print(intervals[1].intersects(intervals[0]))
    print(intervals[2].intersects(intervals[0]))
    for interval in intervals:
        interval.plot(np.array((0, 0)), 1, "tab:gray", 0.4)
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()