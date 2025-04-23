# %%
from __future__ import annotations
from numpy.typing import ArrayLike
import numpy as np
from dataclasses import dataclass, field
import matplotlib as mpl
from typing import Tuple
from matplotlib import pyplot as plt 
from enum import IntEnum
from matplotlib import patches
import numba as nb
from scipy.stats import truncnorm

Position = ArrayLike
Velocity = ArrayLike
Matrix = ArrayLike
Vector = ArrayLike

Angle = float

@nb.njit()
def compute_difference_between_angles(a: Angle, b: Angle) -> Angle:
    """Computes the difference in radians between a and b"""
    if a == b:
        return 0.0

    return np.arccos(np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b))
    #a %= 2 * np.pi
    #b %= 2 * np.pi
    #if np.sign(a - np.pi) == np.sign(b - np.pi):
    #    return a - b
    #else:
    #    return np.abs(a + b - 2 * np.pi)

class State:
    """Models a state in SE(2), that is R^2 x [0, 2pi)"""

    def __init__ (self, pos: Position, angle: Angle):
        """Initializes the state."""
        self.pos = pos 
        self.angle = angle

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

    def __eq__ (self, other: State) -> bool:
        return all(self.pos == other.pos) and self.angle == other.angle

    # NOTE: Not supported with numba
    def __repr__ (self) -> str:
        return f"{(round(float(self.pos[0]), 2), round(float(self.pos[1]), 2), round(self.angle, 2))}"

#@jitclass([
#    ("a", double),
#    ("b", double)
#])
@dataclass
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
            return psi <= self.b or psi >= self.a
        
    def plot(self, ancour: Position, r: float, fillcolor: str, edgecolor: str, alpha: float):
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

        cone = patches.Polygon(points, closed=True, facecolor = fillcolor, edgecolor = edgecolor, alpha = alpha)
        plt.gca().add_patch(cone)

    def generate_uniform_angle(self) -> Angle:
        """Generates a random angle from the angle interval, uniformly."""
        if self.a < self.b:
            return np.random.uniform(self.a, self.b)
        
        # Either generate an angle above or below the x axis depending on the proporition of the 
        # propotion of the arc described by the angle interval being above / below the x axis.
        elif np.random.uniform(0, 1) < self.b / (self.b + 2 * np.pi - self.a):
            return np.random.uniform(0, self.b)
        else:
        #elif np.random.choice(2, size = 1, p = [self.b, 2 * np.pi - self.a]) == 1:
            return np.random.uniform(self.a, 2 * np.pi)
        #else:
            #return np.random.uniform(0, self.b)

    def generate_scewed_uniform_angle(self, mean: float) -> Angle:
        """Generates a scewed uniform angle."""
        a, b = self.a, self.b
        if not (a <= mean <= b):
            raise ValueError("p must be within the interval [a, b].")
 
        # Compute k based on alpha and interval size
        k = 1 / (b - a)
 
        # Sample u from the dynamic uniform range
        u = np.random.uniform(a - mean, b - mean)
 
        # Apply the skewing weight
        weight = np.exp(k * abs(u) / (b - a))
        # Calculate the new angle
        mutation = u * (a - b) * weight
        new_angle = mean + mutation
 
        # Ensure the result stays within [a, b]
        return new_angle % (2 * np.pi) 

    def generate_triangular_angle(self, mean: float) -> Angle:
        """Generates an angle from a triangular distribution"""
        if self.a < self.b:
            return np.random.triangular(self.a, mean , self.b)
        elif self.a > self.b and mean >= self.a:
            return np.random.triangular(self.a, mean, self.b + 2 * np.pi) % (2 * np.pi)
        else:
            return np.random.triangular(self.a, mean + 2 * np.pi, self.b + 2 * np.pi) % (2 * np.pi)

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
#
    # NOTE: Not supported with numba
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

@dataclass()
class AreaOfAcquisition:
    """Models the area of acquisition corresponding to a given angle specification."""
    theta: float
    phi: float

    a: float = field(init = False)
    b: float = field(init = False)

    def __post_init__ (self):
        """Simply initializes a and b"""
        self.a = (self.theta - self.phi) % (2 * np.pi)
        self.b = (self.theta + self.phi) % (2 * np.pi)

    def contains_angle(self, psi: float) -> bool:
        """Checks if the angle is within the area of acquisition"""
        return compute_difference_between_angles(psi, self.theta) <= self.phi

    def generate_uniform_angle(self) -> Angle:
        """Generates a random angle from the angle interval, uniformly."""
        if self.a < self.b:
            return np.random.uniform(self.a, self.b)
        
        # Either generate an angle above or below the x axis depending on the proporition of the 
        # propotion of the arc described by the angle interval being above / below the x axis.
        elif np.random.uniform(0, 1) < self.b / (self.b + 2 * np.pi - self.a):
            return np.random.uniform(0, self.b)
        else:
        #elif np.random.choice(2, size = 1, p = [self.b, 2 * np.pi - self.a]) == 1:
            return np.random.uniform(self.a, 2 * np.pi)

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

    def plot(self, ancour: Position, r_min: float, r_max: float, color: str, alpha: float):
        """Plots the area of acquisition"""
        if self.phi == 3.142:
            plt.gca().add_patch(mpl.patches.Annulus(ancour, r_max, r_max - r_min,  facecolor= color, edgecolor = color, alpha = alpha))
        else:
            if self.a < self.b:
                angles = np.linspace(self.a, self.b, 50) 
            else:
                angles = np.linspace(self.a, (2 * np.pi) * (self.a // (2 * np.pi) + 1) + self.b, 50)

            points_on_outer_arc = np.vstack((ancour[0] + r_max * np.cos(angles),  
                                             ancour[1] + r_max * np.sin(angles)))
        
            points_on_inner_arc = np.vstack((ancour[0] + r_min * np.cos(angles[::-1]),  
                                             ancour[1] + r_min * np.sin(angles[::-1])))

            points = np.vstack([points_on_inner_arc.T, points_on_outer_arc.T])

            cone = patches.Polygon(points, closed=True, facecolor = color, edgecolor = color, alpha = alpha)
            plt.gca().add_patch(cone)

class Dir(IntEnum):
    """Models a direction of a turn"""
    R = 0
    L = 1
    S = 2

    def __repr__ (self) -> str:
        """Return a string representation of the direction."""
        if self == Dir.R: return "R"
        if self == Dir.L: return "L"
        else: return "S"

    def flip (self) -> Dir:
        """Flips the direction."""
        if self == Dir.R: return Dir.L
        elif self == Dir.L: return Dir.R
        else: return Dir.S

  

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
        angles.append(interval.generate_triangular_angle(5.5))
    plt.hist(angles, bins = 30)
    plt.plot()

# %%
