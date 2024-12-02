#%%
# This script converts the TOP problems in the rosources folder to instances Dubins Team Orientering Problem with Angle Depdendent Scores (DOPADS)
import numpy as np 
import os 
from classes.data_types import AngleInterval
from classes.problem_instances.cedopads_instances import CEDOPADSInstance
import random

NUMBER_OF_RELATED_INSTANCES = 1 
path_to_instances = os.path.join(os.getcwd(), "resources", "OP")
for idx, file_id in enumerate(os.listdir(path_to_instances)):
    for instance_number in range(NUMBER_OF_RELATED_INSTANCES):
        print(instance_number)
        with open(os.path.join(path_to_instances, file_id), "r") as file:
            lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

        info_and_depots = lines[:3]
        new_lines = []
        for (x, y, score) in map(lambda line: map(float, line.split()), lines[3:]):
            ell = random.choices([1, 2, 3], weights = [0.2, 0.45, 0.35], k = 1)[0]

            # Angles 
            thetas = [np.random.uniform(0, 2 * np.pi) for _ in range(ell)]
            phis = [np.pi * (np.random.uniform(0.2, 0.8) ** 2) for _ in range(ell)]
            zetas = [phi + np.random.exponential(size=1)[0] for phi in phis]
            assert all([phi <= zeta for phi, zeta in zip(phis, zetas)])

            # Check for overlap and remove indicies if needed
            Js = [AngleInterval(theta - phi, theta + phi) for theta, phi in zip(thetas, phis)]

            indicies_to_remove = []
            for i in range(ell): 
                for j in range(i + 1, ell):
                    if i in indicies_to_remove:
                        continue

                    if (j not in indicies_to_remove) and (Js[i].intersects(Js[j]) or Js[j].intersects(Js[i])):
                        indicies_to_remove.append(j)

            for i in reversed(indicies_to_remove):
                thetas.pop(i)
                phis.pop(i)
                zetas.pop(i)

            # New lines for the problem instance.
            new_lines.append(f"{x} {y} {score}:")
            for theta, phi, zeta in zip(thetas, phis, zetas):
                new_lines[-1] += (f"{round(theta, 3)} {round(phi, 3)} {round(zeta, 3)},")
            new_lines[-1] = new_lines[-1][:-1] # Remove the final comma

        with open(os.path.join(os.getcwd(), "resources", "CEDOPADS", file_id.split(".")[0] + f".{instance_number}.txt"), "w+") as file:
            file.write("\n".join(info_and_depots + new_lines))
        
problem_instance = CEDOPADSInstance.load_from_file("p4.0.txt", needs_plotting = True)
problem_instance.plot(1, [], show = True)