# %%
import numpy as np
import os 
import matplotlib.pyplot as plt 
from typing import Tuple, List
import matplotlib as mpl
from numpy.typing import ArrayLike
from matplotlib import patches

Position = ArrayLike
Velocity = ArrayLike
Matrix = ArrayLike
Vector = ArrayLike

Angle = float

class AngleInterval:

    def __init__ (self, a: float, b: float):
        """Used to model an angle interval between the angles a and b."""
        self.a = a % (2 * np.pi)
        self.b = b % (2 * np.pi)

    def plot(self, ancour: Position, r_max: float, r_min: float, fillcolor: str, edgecolor: str, alpha: float):
        """Plots the angle as a circle arc of radius r with a center at the ancour point"""
        # The angles from the centers 
        #a0, b0 = (self.a + np.pi) % (2 * np.pi), (self.b + np.pi) % (2 * np.pi)
        if self.a < self.b:
            angles = np.linspace(self.a, self.b, 50) 
        else:
            angles = np.linspace(self.a, (2 * np.pi) * (self.a // (2 * np.pi) + 1) + self.b, 50)

        points_on_outer_arc = np.vstack((ancour[0] + r_max * np.cos(angles),  
                                         ancour[1] + r_max * np.sin(angles)))
        
        points_on_inner_arc = np.vstack((ancour[0] + r_min * np.cos(angles[::-1]),  
                                         ancour[1] + r_min * np.sin(angles[::-1])))

        points = np.vstack([points_on_inner_arc.T, points_on_outer_arc.T])

        cone = patches.Polygon(points, closed=True, facecolor = fillcolor, edgecolor = edgecolor, alpha = alpha)
        plt.gca().add_patch(cone)

mpl.use("pgf")

plt.rcParams['figure.figsize'] = [7, 7]
np.random.seed(10)

def generate_obstacles(points: Matrix, lmbda: int, sensing_radius: float, a: float, b: float, R_max: float, R_min: float) -> Tuple[Matrix, Vector]:
    """Generates a set of lmbda valid obsticles, given the parameters."""
    x_max, x_min = np.max(points[:, 0]), np.min(points[:, 0])
    y_max, y_min = np.max(points[:, 1]), np.min(points[:, 1])
    centers, radii = [], []
    while len(centers) != lmbda:
        x = np.random.uniform(x_min - R_max - sensing_radius, x_max + R_max + sensing_radius)
        y = np.random.uniform(y_min - R_max - sensing_radius, y_max + R_max + sensing_radius)
        c = np.array([[x, y]])
        R = np.random.beta(a, b, size=(1, 1)) * (R_max - R_min) + R_min

        if not any(np.linalg.norm(c - p) <= R for p in points):
            centers.append(c)
            radii.append(R)

    return (np.concat(centers), np.concat(radii))

def obs(p: Position, obstacles: List[Tuple[Position, float]], sensing_radius: float, psi: Angle) -> bool:
    """Checks if p is observable with repsect to the obstacles from the angle psi"""
    offset_dir = np.array([np.cos(psi), np.sin(psi)])
    for (c, R) in obstacles:
        # Compute the minimum distance d from the line L_{p, \psi} to c
        t = max(0, min(sensing_radius, np.dot(c - p, offset_dir)))
        restricted_proj = p + t * offset_dir
        d = np.linalg.norm(restricted_proj - c)
        if d <= R:
            return False
        
    return True

def generate_angle_specifications(p: Position, centers: Matrix, radii: Vector, sensing_radius: float, m: int = 1000) -> List[Tuple[float, float]]:
    """Generates a list of valid angle intervals for the point"""
    # Generate O hat.
    O_hat = list(filter(lambda o: np.linalg.norm(o[0] - p) < sensing_radius + o[1], zip(centers, radii)))

    if len(O_hat) == 0: 
        return [(0, np.pi)]

    Psi = []
    for i in range(m):
        if obs(p, O_hat, sensing_radius, 2 * np.pi * i / m) != obs(p, O_hat, sensing_radius, 2 * np.pi * (i + 1) / m):
            Psi.append(np.pi * (2 * i + 1) / m)

    assert len(Psi) % 2 == 0

    if len(Psi) == 0:
        return [(np.random.uniform(0, 2*np.pi), np.pi)] if obs(point, O_hat, sensing_radius, 0) else []

    if obs(p, O_hat, sensing_radius, 0):
        return ([((Psi[2 * i + 1] + Psi[2 * (i + 1)]) / 2, (Psi[2 * (i + 1)] - Psi[2 * i + 1]) / 2) for i in range(int(len(Psi) / 2) - 1)] + 
                [(((2 * np.pi + Psi[0] + Psi[-1]) / 2) % (2 * np.pi), (2 * np.pi + Psi[0] - Psi[-1]) / 2)])
    else:
        return [((Psi[2 * i] + Psi[2 * i + 1]) / 2, (Psi[2 * i + 1] - Psi[2 * i]) / 2) for i in range(int(len(Psi) / 2))] 

path_to_TOP_instances = os.path.join(os.getcwd(), "resources", "TOP")
path_to_CEADTOP_instances = os.path.join(os.getcwd(), "resources", "CEADTOP")
path_to_CEADOP_instances = os.path.join(os.getcwd(), "resources", "CEADOP")

total_distances = [
    [50 + 10 * i for i in range(int((120 - 25)/ 5) + 1)],
    [10 + 10 * i for i in range(int((65 - 5) / 5) + 1)],
    [15 + 5 * i for i in range(14)],
    [20 + 20 * i for i in range(20)]
]

sensing_radii = [
    {"r_min": 1.5, "r_max": 2},
    {"r_min": 0.75, "r_max": 1}, 
    {"r_min": 0.4, "r_max": 0.65},
    {"r_min": 3, "r_max": 4},
]
parameters_for_distribituions = [
    {"lmbda": 200, "a": 2, "b": 4, "R_max": 1.5, "R_min": 0.5},
    {"lmbda": 300, "a": 2, "b": 4, "R_max": 0.75, "R_min": 0.33},
    {"lmbda": 400, "a": 2, "b": 3, "R_max": 0.65, "R_min": 0.11},
    {"lmbda": 500, "a": 1, "b": 3, "R_max": 4, "R_min": 1},
]

print([(parameters_for_distribituions[idx]["R_max"], parameters_for_distribituions[idx]["R_min"]) for idx in range(4)])

# NOTE: We actually load the TOP versions of the problems, but we treat them as OP problems.
for idx, file_id in enumerate(["p4.2.a.txt", "p5.2.a.txt", "p6.2.a.txt", "p7.2.a.txt"]):
    # Load data 
    np.random.seed(idx)
    with open(os.path.join(path_to_TOP_instances, file_id), "r") as file:
        lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

    # Positions and scores
    positions = np.array([[float(line.split()[0]), float(line.split()[1])] for line in lines[4:-1]])
    scores = np.array([float(line.split()[-1]) for line in lines[4:-1]])
    N = positions.shape[0]

    sensing_radius = sensing_radii[idx]["r_max"] 
    centers, radii = generate_obstacles(positions, parameters_for_distribituions[idx]["lmbda"], sensing_radius, a = parameters_for_distribituions[idx]["a"], b = parameters_for_distribituions[idx]["b"], R_max = parameters_for_distribituions[idx]["R_max"], R_min = parameters_for_distribituions[idx]["R_min"])
    X = lines[3]
    for i, point in enumerate(positions):
        # Generate angle intervals
        angle_specifications = generate_angle_specifications(point, centers, radii, sensing_radius)
        
        angles = "".join([f" {theta:.3f} {phi:.3f}" for (theta, phi) in angle_specifications])
        X = "\n".join((X, f"{lines[4 + i]}{angles}"))

    X = "\n".join((X, lines[-1]))

    # Plot everything that is descriped by X, just to check that everything works as expected
    new_lines = X.split("\n")

    plt.title(file_id.split(".")[0])
    plt.scatter(float(new_lines[0].split(" ")[0]), float(new_lines[0].split(" ")[1]), marker="s", color = "black", s = 100, zorder=2)
    plt.scatter(float(new_lines[-1].split(" ")[0]), float(new_lines[-1].split(" ")[1]), marker="d", color = "black", s = 100, zorder = 2)

    # Plot points and angle intervals
    for line in new_lines[1:-1]:
        info = [float(x) for x in line.split(" ")]
        plt.scatter(info[0], info[1], s = 2 * (info[2] + 10), color = "black", zorder=2)

        for jdx in range(1, int(len(info) / 2)):
            phi, theta = info[2 * jdx + 1], info[2 * (jdx + 1)]
            if theta != 3.142:
                AngleInterval(phi - theta, phi + theta).plot(np.array([info[0], info[1]]), r_max = sensing_radius, r_min = sensing_radii[idx]["r_min"], fillcolor = "lightgray", edgecolor = "black", alpha = 0.2)
            else:
                plt.gca().add_patch(mpl.patches.Annulus((info[0], info[1]), sensing_radius, sensing_radius - sensing_radii[idx]["r_min"], facecolor="lightgray", edgecolor = "black", alpha = 0.2))

    for center, radius in zip(centers, radii):
        plt.gca().add_patch(plt.Circle(center, radius, color = "dimgray"))

    plt.axis("equal")
    plt.savefig(f"{file_id.split(".")[0]}.pgf")
    plt.savefig(f"{file_id.split(".")[0]}.pdf")
    plt.clf()

    # Save files
    for m in range(1, 4 + 1):
        distances = [dist / m for dist in total_distances[idx]]
        for jdx, t_max in enumerate(distances):
            info = f"n {N + 2}\nm {m}\ntmax {t_max:.1f}\n" + X
            with open(os.path.join(path_to_CEADTOP_instances, f"{file_id.split(".")[0]}.{m}.{chr(ord("a") + jdx)}.txt"), "w+") as new_file: 
                new_file.write(info)

    for jdx, t_max in enumerate(total_distances[idx]):
        info = f"n {N + 2}\ntmax {t_max:.1f}\n" + X
        with open(os.path.join(path_to_CEADOP_instances, f"{file_id.split(".")[0]}.{m}.{chr(ord("a") + jdx)}.txt"), "w+") as new_file: 
            new_file.write(info)


