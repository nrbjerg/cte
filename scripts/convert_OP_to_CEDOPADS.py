# This script converts the TOP problems in the rosources folder to instances Dubins Team Orientering Problem with Angle Depdendent Scores (DOPADS)
import numpy as np 
import os 
from classes.data_types import AngleInterval
from classes.problem_instances.cedopads_instance import CEDOPADS
import random

path_to_instances = os.path.join(os.getcwd(), "resources", "OP")
for idx, file_id in enumerate(os.listdir(path_to_instances)[:1]):
    np.random.seed(idx)
    with open(os.path.join(path_to_instances, file_id), "r") as file:
        lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

    info_and_depots = lines[:3]
    new_lines = []
    for (x, y, score) in map(lambda line: map(float, line.split()), lines[3:]):
        ell = random.choices([1, 2, 3], weights = [0.2, 0.45, 0.35], k = 1)[0]

        # Angles 
        thetas = [np.random.uniform(0, 2 * np.pi) for _ in range(ell)]
        phis = [np.pi * (np.random.uniform(0.2, 0.8) ** 2) for _ in range(ell)]
        zetas = [1 + np.random.exponential() for _ in range(ell)]

        # Check for overlap and remove indicies if needed
        Js = [AngleInterval(theta - phi, theta + phi) for theta, phi in zip(thetas, phis)]

        indicies_to_remove = []
        for i in range(ell): 
            for j in range(i + 1, ell):
                print(i, j, ell)
                if i in indicies_to_remove:
                    continue

                if (j not in indicies_to_remove) and (Js[i].intersects(Js[j]) or Js[j].intersects(Js[i])):
                    indicies_to_remove.append(j)

        for i in reversed(indicies_to_remove):
            print(i)
            thetas.pop(i)
            phis.pop(i)
            zetas.pop(i)
        
        # New lines for the problem instance.
        new_lines.append(f"{x} {y} {score}:")
        for theta, phi, zeta in zip(thetas, phis, zetas):
            new_lines[-1] += (f"{round(theta, 3)} {round(phi, 3)} {round(zeta, 3)},")
        new_lines[-1] = new_lines[-1][:-1] # Remove the final comma

    print(new_lines)

    with open(os.path.join(os.getcwd(), "resources", "CEDOPADS", file_id), "w+") as file:
        file.write("\n".join(info_and_depots + new_lines))
        
problem_instance = CEDOPADS.load_from_file("set_64_1_60.txt", needs_plotting = True)
problem_instance.plot(r = 1, show = True)