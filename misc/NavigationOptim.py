# %%
import numpy as np
from scipy.optimize import minimize
import random
from tqdm import tqdm
#from p_tqdm import p_umap
from time import perf_counter
import matplotlib.pyplot as plt
 
#t0 = perf_counter()
# Define the function A(x, y)
def A(x, y, **kwargs):
    mu_x, mu_y = 0.5, 0.5
    sigma = 0.2
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma**2))
 
 
def B(x, y, sigma=0.2, mu_x1=0.2, mu_y1=0.2, mu_x2=0.8, mu_y2=0.2, mu_x3=0.2, mu_y3=0.8, mu_x4=1.1, mu_y4=1.1, **kwargs):
    var = 2 * sigma**2
    q = np.exp(-((x - mu_x1) ** 2 + (y - mu_y1) ** 2) / var)
    q += np.exp(-((x - mu_x2) ** 2 + (y - mu_y2) ** 2) / var)
    q += np.exp(-((x - mu_x3) ** 2 + (y - mu_y3) ** 2) / var)
    q += np.exp(-((x - mu_x4) ** 2 + (y - mu_y4) ** 2) / var)
    return q
 

def line_integral(f, x1, y1, x2, y2, num_points=25, sigma=0.2):
    dx = x2 - x1
    dy = y2 - y1
    if num_points is None:
        num_points = int((dx**2 + dy**2) ** 0.5 * 25)
        if num_points < 2:
            num_points = 2
 
    
    x1 += t * dx
    y1 += t * dy
 
    values = f(x1, y1, sigma=sigma)
 
    dx /= num_points - 1
    dy /= num_points - 1
    ds = (dx**2 + dy**2) ** 0.5
 
    integral = 1 / (2 * np.pi * sigma**2) * np.sum(values) * ds
    return integral
 
 
def evolutionary_path_optimizer(A=B, population_size=100, generations=500, mutation_rate=0.01, path_length=20, start=(0, 0), end=(1, 1), BBox=[(0, 0), (1, 1)]):
 
    def generate_path():
        return [start] + [(random.uniform(BBox[0][0], BBox[1][0]), random.uniform(BBox[0][1], BBox[1][1])) for _ in range(path_length - 2)] + [end]
 
    def path_integral(path):
        return sum(abs(line_integral(A, path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])) for i in range(len(path) - 1))
 
    population = [generate_path() for _ in range(population_size)]
    best_paths = []
    best_paths_fitness = []
 
    fitness = [path_integral(path) for path in population]
    fitness_idx = np.argsort(fitness)
    parents = [population[i] for i in fitness_idx[: population_size // 2]]
    for generation in tqdm(range(generations), desc="Generations"):
        next_generation = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(1, path_length - 2)
            child = parent1[:crossover_point] + parent2[crossover_point:]
 
            if random.random() < mutation_rate:
                mutation_point = random.randint(1, path_length - 2)
                child[mutation_point] = (
                    min(max(child[mutation_point][0] + random.uniform(-0.01, 0.01), 0), 1),
                    min(max(child[mutation_point][1] + random.uniform(-0.01, 0.01), 0), 1),
                )
 
            next_generation.append(child)
 
        population = next_generation
        fitness = [path_integral(path) for path in population]
        #fitness = list(p_umap(path_integral, population))
        fitness_idx = np.argsort(fitness)
        parents = [population[i] for i in fitness_idx[: population_size // 2]]
 
        best_path = population[fitness_idx[0]]
        best_paths.append(best_path)
        best_paths_fitness.append(fitness[fitness_idx[0]])
 
    # with open("best_paths.txt", "w") as f:
    #     for generation, path in enumerate(best_paths):
    #         f.write(f"Generation {generation + 1}:\n")
    #         for point in path:
    #             f.write(f"{point[0]}, {point[1]}\n")
    #         f.write("\n")
 
    best_path = best_paths[np.argmin(best_paths_fitness)]
 
    return best_path, BBox, A
 







# %%
plot = True
t = np.linspace(0, 1, 25)

if __name__ == '__main__':
    best_path, BBox, sensing_function = evolutionary_path_optimizer(B, population_size=100, generations=100, mutation_rate=0.01, path_length=15, start=(0, 0), end=(0.5, 0.5), BBox=[(-1, -1), (2, 2)])

    if plot == True:
        x = np.linspace(BBox[0][0], BBox[1][0], 100)
        y = np.linspace(BBox[0][1], BBox[1][1], 100)
        X, Y = np.meshgrid(x, y)
        Z = sensing_function(X, Y)
    
        plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(label="A(x, y)")
    
        best_path_x, best_path_y = zip(*best_path)
        plt.plot(best_path_x, best_path_y, "b-", linewidth=2, label="Best Path")
    
        plt.scatter(*zip(*best_path), c="blue")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Paths and Function A(x, y)")
        plt.show()
 
# %%
##############################################################################################################
# TEST AGAINST MANUAL PATH INTEGRAL
 
 
# def manual_path_integral(f, path):
#     integral = 0
#     for i in range(len(path) - 1):
#         x1, y1 = path[i]
#         x2, y2 = path[i + 1]
#         integral += np.abs(line_integral(f, x1, y1, x2, y2))
#     return integral
 
 
# path = [(0, 0), (0.5, 0), (0.5, 0.5)]
 
# print(manual_path_integral(B, path))
# print(manual_path_integral(B, best_path))
#print(perf_counter() - t0)