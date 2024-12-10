# %%
from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from matplotlib import pyplot as plt 
from matplotlib import patches
import random
from scipy.stats import truncnorm

Position = ArrayLike
Velocity = ArrayLike
Matrix = ArrayLike

Angle = float

def compute_difference_between_angles(a: Angle, b: Angle) -> Angle:
    """Computes the difference in radians between a and b"""
    return np.arccos(np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b))
    #a %= 2 * np.pi
    #b %= 2 * np.pi
    #if np.sign(a - np.pi) == np.sign(b - np.pi):
    #    return a - b
    #else:
    #    return np.abs(a + b - 2 * np.pi)

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

    def plot(self, color = "tab:orange"):
        """Plots the state using matplotlib."""
        plt.scatter(*self.pos, c=color)
        plt.quiver(*self.pos, np.cos(self.angle), np.sin(self.angle), color = color, width = 0.005)

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
        #a0, b0 = (self.a + np.pi) % (2 * np.pi), (self.b + np.pi) % (2 * np.pi)
        if self.a < self.b:
            angles = np.linspace(self.a, self.b, 50) 
        else:
            angles = np.linspace(self.a, (2 * np.pi) * (self.a // (2 * np.pi) + 1) + self.b, 50)

        points_on_arc = np.vstack((ancour[0] + r * np.cos(angles),  
                                   ancour[1] + r * np.sin(angles)))
        points = np.vstack([points_on_arc.T, ancour])

        cone = patches.Polygon(points, closed=True, color = color, alpha = alpha)
        plt.gca().add_patch(cone)

    def generate_uniform_angle(self) -> Angle:
        """Generates a random angle from the angle interval, uniformly."""
        if self.a < self.b:
            return np.random.uniform(self.a, self.b)
        
        # Either generate an angle above or below the x axis depending on the proporition of the 
        # propotion of the arc described by the angle interval being above / below the x axis.
        elif random.choices([0, 1], weights = [self.b, 2 * np.pi - self.a], k = 1)[0] == 1:
            return np.random.uniform(self.a, 2 * np.pi)
        else:
            return np.random.uniform(0, self.b)

    def generate_truncated_normal_angle(self, mean: float, scale_modifier: float = 0.05) -> Angle:
        """Generates a truncated normal angle, with a given mean."""
        # NOTE: Please have a look at the following website https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm
        # for documentation on the truncnorm.rvs method and its arguments.
        if self.a < self.b:
            scale = scale_modifier * (self.b - self.a) 
            return truncnorm.rvs((self.a - mean) / scale, (self.b - mean)  / scale, loc = mean, scale = scale)
        else:
            scale = scale_modifier * (2 * np.pi - self.b + self.a)
            return truncnorm.rvs((self.a - mean)  / scale, (self.b + 2 * np.pi - mean) / scale, loc = mean, scale = scale) % (2 * np.pi)

    def __repr__ (self) -> str:
        """Returns a string representation of the angle interval."""
        return f"[{round(self.a, 3)}; {round(self.b, 3)}]"

    def constrain(self, psi: Angle) -> Angle:
        """Constrain psi to the angle interval."""
        psi %= (2 * np.pi)
        if self.a < self.b and psi < self.a:
            return self.a
        elif self.a < self.b and psi > self.b:
            return self.b
        elif self.a > self.b and psi < self.b:
            return self.b
        elif self.a > self.b and psi > self.a:
            return self.a
        else:
            return psi

if __name__ == "__main__":
    interval = AngleInterval(5, 1)
    angles = []
    eta = np.pi / 2
    
    #for _ in range(100):
        #psi = interval.generate_truncated_normal_angle(5.6)
        #tau = truncnorm.rvs(- eta / 2, eta / 2, loc = np.pi + psi) % (2 * np.pi)
        #point = (np.cos(psi), np.sin(psi))
        #jnterval = AngleInterval(tau - eta / 2, tau + eta / 2)
        #jnterval.plot(point, 1, "tab:blue", 0.1)

        #plt.scatter(*point)
        #plt.quiver(*point, np.cos(tau), np.sin(tau))
    
    #interval.plot((0,0), 1, "tab:orange", 0.2)
    
    for _ in range(100000):
        angles.append(interval.generate_truncated_normal_angle(5.5))
    plt.hist(angles, bins = 200)
    plt.plot()

# %%
