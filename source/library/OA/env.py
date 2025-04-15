import numpy as np 
from dataclasses import dataclass
from typing import List
from classes.data_types import Matrix, Position
import random
import matplotlib.pyplot as plt

class Environment:
    
    def __init__ (self, num_obstacles: int, width: float, height: float, cell_size: float):
        """Initializes an environment with a random number of obstacles.""" 
        self.obstacles = [np.array([np.random.uniform(0, height), np.random.uniform(0, width)]) for _ in range(num_obstacles)]
        self.num_obstacles = num_obstacles
        self.height = height
        self.width = width
        self.n_width = int(width // cell_size + 1)
        self.n_height = int(height // cell_size + 1)

    def get_initial_detection_matrix (self, detection_prop: float, false_positive_rate: float, minimum_distance: float) -> Matrix:
        """Returns a detection matrix, which indicates weather or not we think that we can traverse an edge."""
        matrix = np.zeros((self.n_height, self.n_width))

        # Generate detections 
        actual_detections = [self.obstacles[i] for i in range(self.num_obstacles) if np.random.uniform(0, 1) < detection_prop]
        false_detections = [np.array([np.random.uniform(0, self.height), np.random.uniform(0, self.width)]) for _ in range(int(np.ceil(false_positive_rate * self.num_obstacles)))]
        
        detections = actual_detections + false_detections
        for i in range(self.n_height):
            for j in range(self.n_width):
                position = (self.height / self.n_height * (i + 1 / 2), self.width / self.n_width * (j + 1 / 2))
                matrix[i, j] = np.min([np.linalg.norm(detection - position) for detection in detections]) 
                
        return matrix

if __name__ == "__main__":
    env = Environment(10, 10, 20, 0.25)
    matrix = env.get_initial_detection_matrix(0.5, 2.0, 1.0)
    plt.imshow(matrix)
    plt.show()