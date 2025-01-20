#%%
# This script converts the TOP problems in the rosources folder to instances Dubins Team Orientering Problem with Angle Depdendent Scores (DOPADS)
import numpy as np 
import os 
from classes.data_types import AngleInterval
from classes.problem_instances.cedopads_instances import CEDOPADSInstance
import random
import matplotlib.pyplot as plt 

NUMBER_OF_RELATED_INSTANCES = 1 
path_to_instances = os.path.join(os.getcwd(), "resources", "OP")
for idx, file_id in enumerate(os.listdir(path_to_instances)):
    with open(os.path.join(path_to_instances, file_id), "r") as file:
        lines = list(map(lambda line: line.replace("\t", " "), file.read().splitlines()))

    info = lines[1]
    

    # Compute risk matrix
    positions = np.array(list(map(lambda line: np.array(list(map(float, line.split()))[:-1]), lines[1:])))
    print(positions.shape)
    N = len(positions)
    m = int(np.floor((N - 2) / 4))

    x_max, x_min = np.max(positions[:, 0]), np.min(positions[:, 0])
    y_max, y_min = np.max(positions[:, 1]), np.min(positions[:, 1])

    plt.scatter(positions[:, 0], positions[:, 1])
    plt.show()