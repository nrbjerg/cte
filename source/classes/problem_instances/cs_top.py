# %%
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Tuple
import numpy as np
import os
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import matplotlib as mpl
from classes.data_types import Position, Matrix, Vector, Angle
from scipy.stats import multivariate_normal

plt.rcParams['figure.figsize'] = [7, 7]

@dataclass
class Sweep:
    """Models a sweep, a plan will consist of a sequence of sweeps"""
    middle: Position
    orientation: Angle
    radii: Tuple[float, float]

@dataclass
class CS_TOPInstance:
    """Stores the data related to a TOP instance."""
    problem_id: str
    number_of_agents: int
    t_max: float

    source: Position
    sink: Position

    means: List[Position]
    covs: List[Matrix]
    weights: Vector

    bounds: Matrix # bounds[0][0] = x_min, bounds[0][1] = x_max, bounds[1][0] = x_min, bounds[1][1] = x_max

    pdf: Callable[[Position], float] = field(init = False)

    @staticmethod
    def generate_random_instance(N: int, bounds: Matrix, number_of_agents: int, t_max: float, problem_id: str = "Randomly Generated Instance"):
        """Generates a random CS_TOP instance, using N bivariate normal distributions."""
        np.random.seed(42)
        means = [np.array([np.random.uniform(bounds[0][0], bounds[0][1]),
                           np.random.uniform(bounds[1][0], bounds[1][1])]) for _ in range(N)]
    
        covs = []
        for _ in range(N):
            sigma_x, sigma_y = 2 + np.random.exponential(1), 2 + np.random.exponential(1) 
            rho = 2 * (np.random.beta(3, 3) - 0.5)
            covs.append(np.array([[sigma_x ** 2, rho * sigma_x * sigma_y], [rho * sigma_x * sigma_y, sigma_y ** 2]]))

        unnormalized_weights = np.random.uniform(0.2, 1, size=N)
        weights = 1 / np.sum(unnormalized_weights) * unnormalized_weights

        source = np.array([np.random.uniform(bounds[0][0], bounds[0][1]),
                           np.random.uniform(bounds[1][0], bounds[1][1])])

        sink = np.array([np.random.uniform(bounds[0][0], bounds[0][1]),
                           np.random.uniform(bounds[1][0], bounds[1][1])])

        return CS_TOPInstance(problem_id, number_of_agents, t_max, source, sink, means, covs, weights, bounds)

    def __post_init__ (self):
        """Initialize the probability map etc."""
        assert np.isclose(sum(self.weights), 1.0)

        self.number_of_samples = 30_000
        self.samples = np.zeros((self.number_of_samples, 2))
        
        for i in range(self.number_of_samples):
            j = np.random.choice(len(self.means), p = self.weights)
            while True: # Resample until the point lies within the rectangle defined by the bounds.
                sample = multivariate_normal.rvs(mean=self.means[j], cov = self.covs[j])
                if (sample[0] >= self.bounds[0][0] and sample[0] <= self.bounds[0][1] and
                    sample[1] >= self.bounds[1][0] and sample[1] <= self.bounds[1][1]):
                    self.samples[i] = sample
                    break

        self.pdf = lambda x: sum(w * multivariate_normal.pdf(x, mean=mu, cov=sigma) for w, mu, sigma in zip(self.weights, self.means, self.covs))

    def plot(self, number_of_contour_lines: int = 10, show: bool = False, plot_means: bool = False):
        """Plots the PDF coresponding to the means and covs specified within the instance."""
        plt.style.use("ggplot")

        #xs = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
        #ys = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
        #zs = np.array([[self.pdf([x, y]) for x in xs] for y in ys])
        #plt.contour(xs, ys, zs, levels=number_of_contour_lines, cmap="Reds", zorder=2)
        plt.scatter(*self.source, 120, marker = "s", c = "black", zorder=4)
        plt.scatter(*self.sink, 120, marker = "D", c = "black", zorder=4)

        plt.scatter(self.samples[:, 0], self.samples[:, 1], color = "black", alpha = 0.1)
        if plot_means:
            plt.scatter([mean[0] for mean in self.means], [mean[1] for mean in self.means], self.weights * 640, color = "red", zorder=3)

        if show:
            plt.show()


def main():
    import time 
    cs_top = CS_TOPInstance.generate_random_instance(128, np.array([[0, 30], [0, 30]]), 2, 3)
    start_time = time.time()
    cs_top.plot(show = True, plot_means = True)
    print(f"Took at total of {time.time() - start_time}")


if __name__ == "__main__":
    main()
        