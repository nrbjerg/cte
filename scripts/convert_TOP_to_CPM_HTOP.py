#!/usr/bin/env python3
import numpy as np
from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
import os
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
import matplotlib as mpl
from tqdm import tqdm

path_to_TOP_instances = os.path.join(os.getcwd(), "resources", "TOP")
path_to_CPM_RTOP_instances = os.path.join(os.getcwd(), "resources", "CPM_RTOP")

plotted = False

for idx, file_id in tqdm(enumerate(sorted(os.listdir(path_to_TOP_instances)))):
    # Load data 
    np.random.seed(idx)
    with open(os.path.join(path_to_TOP_instances, file_id), "r") as file:
        lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

    info = lines[:3]

    # Compute risk matrix
    positions = list(map(lambda line: np.array(list(map(float, line.split()))[:-1]), lines[3:]))
    scores = list(map(lambda line: float(line.split()[-1]), lines[4:-1]))
    N = len(positions)
    m = int(np.round((N - 2) / 3))
    print(m)

    x_max, x_min = max(map(lambda p: p[0], positions)), min(map(lambda p: p[0], positions))
    y_max, y_min = max(map(lambda p: p[1], positions)), min(map(lambda p: p[1], positions))

    means = np.vstack((np.random.uniform(x_min, x_max, size=m), np.random.uniform(y_min, y_max, size=m)))

    sigma_xs, sigma_ys = (x_max - x_min) / 8 * np.random.uniform(0.5, 1, size = m), (y_max - y_min) / 8 * (np.random.uniform(0.5, 1, size = m))  

    #sigma_xs, sigma_ys = np.random.exponential(2, 0.7, size=m), np.random.lognormal(2, 0.7, size = m)
    rhos = 2 * (np.random.beta(3, 3, size = m) - 0.5) 

    covariances = [np.array([[sigma_x ** 2, rho * sigma_x * sigma_y],
                             [rho * sigma_x * sigma_y, sigma_y ** 2]]) 
                   for sigma_x, sigma_y, rho in zip(sigma_xs, sigma_ys, rhos)]

    distributions = [multivariate_normal(mean, cov) for mean, cov in zip(means.T, covariances)]

    f = lambda x: sum(distribution.pdf(x) for distribution in distributions)

    # Plot the contour plot of a single f:
    if not plotted:
        plt.rcParams['figure.figsize'] = [12, 9]
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.gca().set_aspect("equal", adjustable="box")
        xs = np.linspace(-0.5, 30.5, 200)
        ys = np.linspace(-0.5, 30.5, 200)
        zs = np.array([[f([x, y]) for x in xs] for y in ys])
        c_map = mpl.colors.LinearSegmentedColormap.from_list("", ["tab:green", "tab:blue", "tab:purple", "tab:red"])
        plt.contour(xs, ys, zs, levels=16, cmap=c_map, zorder=1)
        plt.colorbar(mpl.cm.ScalarMappable(norm=plt.Normalize(0, 0.084), cmap=c_map), ax=plt.gca())
        min_score = min(scores) 
        max_score = max(scores) 
        for pos, s in zip(positions[1:-1], scores):
            plt.scatter(pos[0], pos[1], (0.2 + (s - min_score) / (max_score - min_score)) * 120, color="tab:gray", zorder=2)

        plt.scatter(*positions[0], 120, marker = "s", c = "black", zorder=10)
        plt.scatter(*positions[-1], 120, marker = "d", c = "black", zorder=10)
        plt.title(f"TOP Instance: {file_id[:-4]}")
        plotted = True
        plt.ylim(-0.5, 30.5)
        plt.xlim(-0.5, 30.5)
        plt.savefig("from_generation.png", bbox_inches="tight")

    omegas = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            if i >= j or all(positions[i] == positions[j]): continue

            dist = np.linalg.norm(positions[i] - positions[j])
            ell = int(np.ceil(dist / 2)) 
            if ell == 0:
                raise ValueError("Got ell == 0")

            phi = lambda s: positions[i] * s + positions[j] * (1 - s)
            omegas[i, j] = dist / ell * sum(f(phi(k / (ell + 1))) for k in range(1, ell + 1))
            omegas[j, i] = omegas[i, j]

    omega_max = max(max(omegas[i, j] for i in range(N) if i < j) for j in range(N) if j != 0)
    rs = 1 / (4 * omega_max) * omegas

    # Save data 
    points_and_risks = "\n".join([line + " " + str(list(map(lambda r: float(round(r, 4)), rs[i])))[1:-1].replace(",", "") for i, line in enumerate(lines[3:])])

    heat_map = np.zeros((100, 100))
    for i, x in enumerate(np.linspace(x_min, x_max, 100)):
        for j, y in enumerate(np.linspace(y_min, y_max, 100)):
            heat_map[i, j] = f([x, y])

    t_max = float(info[-1].split(" ")[-1])

    for idx, velocity in enumerate([1.4, 1.6, 1.8]):
        for jdx, distance in enumerate([0.6 * t_max, 0.7 * t_max, 0.8 * t_max]):
            cpm_info = f"\nVcpm {velocity}\ndmax {distance}\n"

            new_file_id = ".".join(file_id.split(".")[:-1]) + f".{idx * 3 + jdx}.txt"
            with open(os.path.join(path_to_CPM_RTOP_instances, new_file_id), "w+") as new_file:
                new_info = "\n".join(info) + cpm_info 
                new_file.write(new_info + points_and_risks)