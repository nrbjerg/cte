#%%
#!/usr/bin/env python3
import numpy as np
from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
import gstools as gs
import os
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
from tqdm import tqdm

path_to_TOP_instances = os.path.join(os.getcwd(), "resources", "TOP")
path_to_CPM_HTOP_instances = os.path.join(os.getcwd(), "resources", "CPM_HTOP")

for idx, file_id in tqdm(enumerate(list(os.listdir(path_to_TOP_instances)))):
    # Load data 
    np.random.seed(idx)
    with open(os.path.join(path_to_TOP_instances, file_id), "r") as file:
        lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

    info = lines[:3]

    # Compute risk matrix
    positions = list(map(lambda line: np.array(list(map(float, line.split()))[:-1]), lines[3:]))
    N = len(positions)
    m = int(np.floor((N - 2) / 4))

    x_max, x_min = max(map(lambda p: p[0], positions)), min(map(lambda p: p[0], positions))
    y_max, y_min = max(map(lambda p: p[1], positions)), min(map(lambda p: p[1], positions))

    
    means = np.vstack((np.random.uniform(x_min, x_max, size=m), np.random.uniform(y_min, y_max, size=m)))

    sigma_xs, sigma_ys = np.random.lognormal(2, 0.7, size=m), np.random.lognormal(2, 0.7, size = m)
    rhos = np.random.uniform(-0.5, 0.5, size = m)

    covariances = [np.array([[sigma_x ** 2, rho * sigma_x * sigma_y],
                             [rho * sigma_x * sigma_y, sigma_y ** 2]]) 
                   for sigma_x, sigma_y, rho in zip(sigma_xs, sigma_ys, rhos)]

    distributions = [multivariate_normal(mean, cov) for mean, cov in zip(means.T, covariances)]

    f = lambda x: sum(distribution.pdf(x) for distribution in distributions)

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
    slow, fast = {"V_cpm": 1.4, "kappa": 1}, {"V_cpm": 1.8, "kappa": 1.3}
    short, long = {"dmax": round(1.2 * t_max, 4)}, {"dmax": round(1.4 * t_max, 4)}

    for idx, (speed, distance) in enumerate([(slow, short), (fast, short), (slow, long), (fast, long)]):
        cpm_info = f"\nVmax {speed["Vmax"]}\nVmin {speed["Vmin"]}\nkappa {speed["kappa"]}\ndmax {distance["dmax"]}\n"

        new_file_id = ".".join(file_id.split(".")[:-1]) + f".{idx}.txt"
        with open(os.path.join(path_to_CPM_HTOP_instances, new_file_id), "w+") as new_file:
            new_info = "\n".join(info) + cpm_info 
            new_file.write(new_info + points_and_risks)