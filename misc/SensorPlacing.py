import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
from tqdm import tqdm

import matplotlib.pyplot as plt


class SensorField:
    def __init__(self, area_width, area_height, lambda_cameras, lambda_radars, camera_range, radar_range):
        self.area_width = area_width
        self.area_height = area_height
        self.lambda_cameras = lambda_cameras
        self.lambda_radars = lambda_radars
        self.camera_range = camera_range
        self.radar_range = radar_range
        self.cameras = self.generate_sensors("camera")
        self.radars = self.generate_sensors("radar")

    def generate_sensors(self, sensor_type, n_cameras=None, n_radars=None, method="poisson"):

        if sensor_type == 'camera':
            if n_cameras is None:
                n_sensors = np.random.poisson(self.lambda_cameras * self.area_width * self.area_height)
            else:
                n_sensors = n_cameras
            x   = np.random.uniform(0, self.area_width, (n_sensors, 1))
            y   = np.random.uniform(0, self.area_height, (n_sensors, 1))
            positions = np.concatenate((x, y), axis=1)
            return [Sensor(x, y, sensor_type, self.camera_range) for x, y in positions]
        
        elif sensor_type == 'radar':
            if n_radars is None:
                n_sensors = np.random.poisson(self.lambda_radars * self.area_width * self.area_height)
            else:
                n_sensors = n_radars
            x   = np.random.uniform(0, self.area_width, (n_sensors, 1))
            y   = np.random.uniform(0, self.area_height, (n_sensors, 1))
            positions = np.concatenate((x, y), axis=1)
            return [Sensor(x, y, sensor_type, self.radar_range) for x, y in positions]
        else:
            print('Invalid sensor type. Please use "camera" or "radar".')

    def manual_add_sensor(self, sensor_type, x, y):
        if sensor_type == 'camera':
            self.camera_positions = np.append(self.camera_positions, [[x, y]], axis=0)
        elif sensor_type == 'radar':
            self.radar_positions = np.append(self.radar_positions, [[x, y]], axis=0)
        else:
            print('Invalid sensor type. Please use "camera" or "radar".')

    def manual_remove_sensor(self, sensor_type, x, y): 
        if sensor_type == 'camera':
            self.camera_positions = np.delete(self.camera_positions, np.where((self.camera_positions == [x, y]).all(axis=1)), axis=0)
        elif sensor_type == 'radar':
            self.radar_positions = np.delete(self.radar_positions, np.where((self.radar_positions == [x, y]).all(axis=1)), axis=0)
        else:
            print('Invalid sensor type. Please use "camera" or "radar".')
    def get_positions(self, sensor_type):
        if sensor_type == 'camera':
            return [sensor.get_position() for sensor in self.cameras]
        elif sensor_type == 'radar':
            return [sensor.get_position() for sensor in self.radars]
        else:
            print('Invalid sensor type. Please use "camera" or "radar".')
        
class Sensor:
    def __init__(self, x, y, type, range):
        self.x = x
        self.y = y
        self.type = type
        self.range = range
    
    def detect(self, drone):
        return np.sqrt((self.x - drone.x)**2 + (self.y - drone.y)**2) <= self.range
    
    def get_position(self):
        return (self.x, self.y)
    
class Drone:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def move(self, x, y):
        self.x = x
        self.y = y
    
    def detect(self, sensors):
        return [sensor.detect(self) for sensor in sensors]

def generate_sensor_field_kernel(sensor_field, x, y, variance_camera, variance_radar):
    kernel = np.zeros((len(x), len(y)))
    x_grid, y_grid = np.meshgrid(x, y)

    for sensor in sensor_field.cameras:
        sensor_x, sensor_y = sensor.get_position()
        kernel += np.exp(-((x_grid - sensor_x)**2 + (y_grid - sensor_y)**2) / (2 * variance_camera))

    for sensor in sensor_field.radars:
        sensor_x, sensor_y = sensor.get_position()
        kernel += np.exp(-((x_grid - sensor_x)**2 + (y_grid - sensor_y)**2) / (2 * variance_radar))

    return kernel

def plot_sensor_field_kernel(sensor_field, x, y, variance_camera, variance_radar):
    kernel = generate_sensor_field_kernel(sensor_field, x, y, variance_camera, variance_radar)
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, kernel, levels=50, cmap='viridis')
    plt.colorbar(label='Kernel Intensity')
    plt.scatter(
        [sensor.x for sensor in sensor_field.cameras],
        [sensor.y for sensor in sensor_field.cameras],
        c='red', label='Cameras', edgecolor='black'
    )
    plt.scatter(
        [sensor.x for sensor in sensor_field.radars],
        [sensor.y for sensor in sensor_field.radars],
        c='blue', label='Radars', edgecolor='black'
    )
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Sensor Field Kernel')
    plt.legend()
    plt.show()



#%%
class SensorPlacementGeneticAlgorithm:
    def __init__(self, environment, population_size, generations, mutation_rate, n_cameras=None, n_radars=None):
        self.n_cameras = n_cameras
        self.n_radars = n_radars
        self.environment = environment
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            cameras = self.environment.generate_sensors("camera", n_cameras=self.n_cameras)
            radars = self.environment.generate_sensors("radar", n_radars=self.n_radars)
            population.append((cameras, radars))
        return population

    def fitness(self, individual):
        cameras, radars = individual
        total_coverage = 0
        drones = [Drone(0, 0)]  # Starting positions of the drones
        paths = [
            [(x, np.sin(x / 2) * 2 + 2.5) for x in np.linspace(0, self.environment.area_width, 100)],
            
        ]
        
        for drone, path in zip(drones, paths):
            for (x, y) in path:
                drone.move(x, y)
                if any(sensor.detect(drone) for sensor in cameras + radars):
                    total_coverage += 1
        return total_coverage

    def selection(self):
        sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        crossover_point = len(parent1[0]) // 2
        child1_cameras = parent1[0][:crossover_point] + parent2[0][crossover_point:]
        child1_radars = parent1[1][:crossover_point] + parent2[1][crossover_point:]
        child2_cameras = parent2[0][:crossover_point] + parent1[0][crossover_point:]
        child2_radars = parent2[1][:crossover_point] + parent1[1][crossover_point:]
        return (child1_cameras, child1_radars), (child2_cameras, child2_radars)

    def mutate(self, individual):
        cameras, radars = individual
        if random.random() < self.mutation_rate:
            cameras[random.randint(0, len(cameras) - 1)] = Sensor(
                np.random.uniform(0, self.environment.area_width),
                np.random.uniform(0, self.environment.area_height),
                "camera",
                self.environment.camera_range
            )
        if random.random() < self.mutation_rate:
            radars[random.randint(0, len(radars) - 1)] = Sensor(
                np.random.uniform(0, self.environment.area_width),
                np.random.uniform(0, self.environment.area_height),
                "radar",
                self.environment.radar_range
            )
        return cameras, radars

    def run(self):
        for _ in tqdm(range(self.generations), desc="Running Genetic Algorithm"):
            selected_population = self.selection()
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
            self.population = new_population
        best_individual = max(self.population, key=self.fitness)
        self.environment.cameras, self.environment.radars = best_individual

#%%
# Example usage of plot_sensor_field_kernel
if __name__ == "__main__":
    # Define sensor field parameters
    area_width = 10
    area_height = 10
    lambda_cameras = 0.1
    lambda_radars = 0.1
    camera_range = 2
    radar_range = 3

    # Create a sensor field
    sensor_field = SensorField(area_width, area_height, lambda_cameras, lambda_radars, camera_range, radar_range)

    # Define grid for kernel visualization
    x = np.linspace(0, area_width, 100)
    y = np.linspace(0, area_height, 100)

    # Variance for kernel computation
    variance_camera = 0.3
    variance_radar = 1.5

    # Plot the sensor field kernel
    plot_sensor_field_kernel(sensor_field, x, y, variance_camera, variance_radar)


# %%
