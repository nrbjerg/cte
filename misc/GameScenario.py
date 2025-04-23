
from misc.SensorPlacing import SensorField, Sensor, Drone, SensorPlacementGeneticAlgorithm, plot_sensor_field_kernel, plot_sensor_field_kernel
import numpy as np
import misc.NavigationOptim as nav


# Parameters
area_width = 20
area_height = 5
lambda_cameras = 0.1  # Density of cameras
lambda_radars = 0.07  # Density of radars
camera_range = 1
radar_range = 2

# Create SensorPlacement object
environment = SensorField(area_width, area_height, lambda_cameras, lambda_radars, camera_range, radar_range)


# Access positions
camera_positions = environment.get_positions('camera')
radar_positions = environment.get_positions('radar')
print(camera_positions)
print(radar_positions)

# Parameters for the genetic algorithm
population_size = 50
generations = 100
mutation_rate = 0.1
n_cameras = 10
n_radars= 10
# Run the genetic algorithm
sensor_placement_optimizer = SensorPlacementGeneticAlgorithm(environment, population_size, generations, mutation_rate, n_cameras, n_radars)
sensor_placement_optimizer.run()

# Access optimized positions
camera_positions = environment.get_positions('camera')
radar_positions = environment.get_positions('radar')
# Print positions