# This script converts the TOP problems in the rosources folder to instances Dubins Team Orientering Problem with Angle Depdendent Scores (DTOPADS)
import numpy as np 
import gstools as gs
import os 
import random

model = gs.Gaussian(dim=2, var=1, len_scale=10)
path_to_instances = os.path.join(os.getcwd(), "resources", "TOP")
for idx, file_id in enumerate(os.listdir(path_to_instances)[:1]):
    srf = gs.SRF(model, generator="VectorField", seed = idx)
    
    with open(os.path.join(path_to_instances, file_id), "r") as file:
        lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

    info_and_initial_point = lines[:4]
    new_lines = []
    for (x, y, score) in map(lambda line: map(float, line.split()), lines[4:-1]):
        m = random.choices([1, 2, 3], weights = [0.7, 0.2, 0.1], k = 1)[0]

        # Angles 
        x0, y0 = tuple(srf((x, y)))
        angle = np.atan2(y0, x0)[0]
        angle_offset = random.random() * np.pi * 4 / 5 
        angles = [angle + angle_offset * multiplier for multiplier, _ in zip([0, 1, -1], range(m))]

        # Field of views
        fovs = [3 + random.random() * 3 for _ in range(m)] # will be a number in [3 / 2; 3]

        # Distribute the scores to each angle phi.
        v = np.random.rand(m) 
        v_hat = v / np.linalg.norm(v)
        scores = [score * scalar for scalar in v_hat]
        
        # New lines for the problem instance.
        new_lines.append(f"{x} {y} [ ")
        for angle, fov, score in zip(angles, fovs, scores):
            new_lines[-1] += (f"{round(angle, 2)} {round(fov, 2)} {round(score, 2)} ")
        new_lines[-1] += "]"
            
    with open(os.path.join(os.getcwd(), "resources", "DTOPADS", file_id), "w+") as file:
        file.write("\n".join(info_and_initial_point + new_lines + [lines[-1]]))
        
            
    
    