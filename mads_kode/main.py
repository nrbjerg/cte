# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import import_TOP_instance
import Instance_plot
from scipy.spatial.distance import cdist
import OP_solver
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import cProfile

filepath = "C:/Users/NX83SQ/GitHub/Benchmark_instances/Set_100/p4.4.j.txt"
n, m, tmax = import_TOP_instance.import_numbers(filepath)[0], import_TOP_instance.import_numbers(filepath)[1], import_TOP_instance.import_numbers(filepath)[2]
temp_TOP = import_TOP_instance.parse_top_file(filepath)
TOP = temp_TOP[1:-1]

#Calculating the midpoints of the x- values in the TOP instance, for the feasalbe zone boundries
x_values_sorted = np.sort(TOP[:, 0])
x_vals = [x_values_sorted[0]]
for i in range(len(x_values_sorted) - 1):
    midpoint = (x_values_sorted[i] + x_values_sorted[i + 1]) / 2
    x_vals.append(midpoint)
x_vals.append(x_values_sorted[-1])

# Initialize zone pairs
# zone_pairs is a 2D array where each element is a pair of indices representing [score, counter]
zone_pairs = np.zeros((len(x_vals), len(x_vals), 2))
for i in range(len(x_vals)): # NOTE: (MARTIN) burde kunne slettes?
    for j in range(len(x_vals)):
            zone_pairs[i, j] = [0, 0]


initialXValue = np.linspace(min(TOP[:,0]),max(TOP[:,0]),m+1,endpoint=True) # creates equally spaced vertical lines, including the start and end points, so when used for plotting, use [1:-1] to remove the endpoints
# print(sum(TOP[:,2]))

def initial_splice_by_score(TOP, number_of_units):
    """
    Splicing the TOP instance into a section pr unit, based on the score of the points returning the boundaries of the sections.
    
    TOP: The TOP instance: [[x,y,score],[x,y,score],...]
    number_of_units: The number of units the TOP instance should be split into: int
    """
    points = TOP[np.argsort(TOP[:, 0])]

    #Calculating the total score
    total_score = sum(points[:,2])
    target_score = total_score/number_of_units

    boundaries = []
    accumulated_score = 0

    for i in range(len(points)-1):
        accumulated_score += points[i][2]

        if accumulated_score >= target_score and len(boundaries) < number_of_units - 1: # NOTE: (MARTIN) tror aldrig at den sidste betingelse rammes?
            boundaries.append((points[i][0]+points[i+1][0])/2)  # Use current x as a boundary
            accumulated_score = 0  # Reset score counter
    
    return boundaries

def instance_splice(top_instance, x_values):
    """
    Splicing the TOP instance into sections based on the x_values. Returns a list of sections to be parsed to a solver. Adds a start and sink node to each section as well.

    top_instance: The TOP instance: [[x,y,score],[x,y,score],...]
    x_values: The x values to split the instance by: [x1, x2, x3, ...]
    """
    points = top_instance[np.argsort(top_instance[:, 0])]
    sections = []
    
    # Find the lowest and highest y values in the whole TOP instance
    y_min = np.min(top_instance[:, 1])
    y_max = np.max(top_instance[:, 1])
    
    # NOTE: (MARTIN) Du kan måske fjerne edge cases ved at tilføje x_min og x_max + maskinepsilon til x_values, så tror jeg du kan nøjes med for loopet.
    
    # Add all points before the first x_value in a section
    x_min = x_vals[0]
    x_max = x_vals[x_values[0]]
    section = points[(points[:, 0] >= x_min) & (points[:, 0] < x_max)]
    x_middle_value = (x_min + x_max) / 2
    section = np.vstack(([x_middle_value, y_min, 0], section, [x_middle_value, y_max, 0]))
    sections.append(section)
    
    for i in range(len(x_values) - 1):
        x_min, x_max = x_vals[x_values[i]], x_vals[x_values[i+1]]
        section = points[(points[:, 0] >= x_min) & (points[:, 0] < x_max)]
        #Adding the start and sink node to the section
        x_middle_value = (x_min + x_max) / 2
        section = np.vstack(([x_middle_value, y_min, 0], section, [x_middle_value, y_max, 0]))
        sections.append(section)
    
    # Add all points after the last x_value in a section
    x_min = x_vals[x_values[-1]]
    x_max = x_vals[-1]
    section = points[(points[:, 0] >= x_min) & (points[:, 0] < x_max)]
    x_middle_value = (x_min + x_max) / 2
    section = np.vstack(([x_middle_value, y_min, 0], section, [x_middle_value, y_max, 0]))
    sections.append(section)
    
    return sections

def solve_multiple(spliced_instance, x_values, tmax, alpha = 0.2, max_iter = 100):
    """
    Solving multiple spliced instances and returning the solutions.

    spliced_instance: The spliced instance: [section1, section2, ...]
    tmax: The maximum time allowed for a route: int
    alpha: The alpha value for the GRASP algorithm: float
    max_iter: The maximum number of iterations for the GRASP algorithm: int
    """

    x_vals_check = np.insert(x_values, 0, 0)
    x_vals_check = np.append(x_vals_check, len(x_vals)-1)

    scores = []
    for i in range(len(spliced_instance)):
        if zone_pairs[x_vals_check[i], x_vals_check[i+1], 1] > 3:
            # print("skipped")
            scores.append(zone_pairs[x_vals_check[i], x_vals_check[i+1], 0])
        else:
            solution = OP_solver.GRASP_OP_optimized(spliced_instance[i], tmax, alpha, max_iter)
            score = sum(solution[:, 2])
            scores.append(score)
            if zone_pairs[x_vals_check[i], x_vals_check[i+1], 0] < score:
                zone_pairs[x_vals_check[i], x_vals_check[i+1], 0] = score
            zone_pairs[x_vals_check[i], x_vals_check[i+1], 1] += 1

    total_score = sum(scores)       
    return total_score 

def run_sol_multiple(spliced_instance, x_values, tmax, n, alpha = 0.3, max_iter = 100, no_progress_bar = False):
    scores=[]
    for j in tqdm(range(n), desc="Finding best solution", disable=no_progress_bar):
        score = solve_multiple(spliced_instance, x_values, tmax, alpha, max_iter)
        scores.append(score)
        
    return np.max(scores)

def pso_optimize(top_instance, num_particles, num_iterations, tmax, m, alpha=0.3, max_iter=100, no_progress_bar=False, early_stopping=True):
    
    # Initialize particles
    x_vals_inner = np.array(x_vals[1:-1])  # exclude endpoints
    num_inner = len(x_vals_inner)
    particles = [np.sort(np.random.choice(np.arange(num_inner), size=m-1, replace=False)) for _ in range(num_particles)]
    velocities = [np.zeros(m-1, dtype=float) for _ in range(num_particles)]

    # Initialize best positions
    personal_best_positions = particles.copy()
    personal_best_scores = [evaluate_fitness(p, top_instance, tmax, alpha, max_iter) for p in particles]
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    # PSO parameters
    w_max = 0.9
    w_min = 0.4
    c1 = 1.2
    c2 = 1.6
    max_velocity = (x_vals[-1] - x_vals[0]) / 10

    score_evol = np.zeros((num_particles, num_iterations))
    for j in tqdm(range(num_iterations), desc="Optimizing", disable=no_progress_bar):
        w = w_max - (w_max - w_min) * (j / num_iterations)

        for i, particle in enumerate(particles):
            # Convert index-based particle to x-values
            x_particle = x_vals_inner[particle]
            personal_best_x = x_vals_inner[personal_best_positions[i]]
            global_best_x = x_vals_inner[global_best_position]

            # Update velocity
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_x - x_particle) +
                             c2 * r2 * (global_best_x - x_particle))
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)

            # Update position in x-value space
            new_x_vals = x_particle + velocities[i]
            new_x_vals = np.clip(new_x_vals, x_vals_inner[0], x_vals_inner[-1])

            # NOTE: (MARTIN) Måske giver det mening at flytte partiklen lige meget hvad,
            #       Fordi hvis du gør det du gør her nedenfor kan det være at den bliver stående
            #       Som jeg forstår det ihvertfald.

            # Map back to closest indices in x_vals_inner
            new_indices = np.array([np.argmin(np.abs(x_vals_inner - x)) for x in new_x_vals])
            new_indices = np.unique(new_indices)

            # Ensure correct number of indices
            if len(new_indices) < m - 1:
                available = np.setdiff1d(np.arange(num_inner), new_indices)
                extra_indices = np.random.choice(available, size=(m - 1 - len(new_indices)), replace=False)
                new_indices = np.sort(np.concatenate([new_indices, extra_indices]))
            else:
                new_indices = np.sort(new_indices[:m - 1])

            particles[i] = new_indices

            # Evaluate fitness
            score = evaluate_fitness(particles[i], top_instance, tmax, alpha, max_iter)
            score_evol[i, j] = score

            # Update personal and global bests
            if score > personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = score
                if score > global_best_score:
                    global_best_position = particles[i]
                    global_best_score = score

        # Early stopping
        if j > num_iterations // 3 and early_stopping:
            scores_j = score_evol[:, j]
            mean_j = np.mean(scores_j)
            std_j = np.std(scores_j)
            z = (np.max(scores_j) - mean_j) / (std_j / np.sqrt(len(scores_j))) if std_j > 0 else 0
            if z < 1.75: # z-score threshold for early stopping
                print(f"Early stopping at iteration {j} due to low z-score ({z:.2f})")
                return global_best_position, global_best_score, score_evol

    return global_best_position, global_best_score, score_evol


#def pso_optimize(top_instance, num_particles, num_iterations, tmax, alpha=0.3, max_iter=100, no_progress_bar=False, early_stopping=True):
    # Initialize particles
    x_min = np.min(top_instance[:, 0])
    x_max = np.max(top_instance[:, 0])
    m = top_instance.shape[0]
    particles = [np.sort(np.random.uniform(x_min, x_max, size=m-1)) for _ in range(num_particles)]
    velocities = [np.zeros_like(p) for p in particles]

    # Initialize best positions
    personal_best_positions = particles.copy()
    personal_best_scores = [evaluate_fitness(p, top_instance, tmax, alpha, max_iter) for p in particles]
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    # PSO parameters
    w_max = 0.9
    w_min = 0.4
    c1 = 1.2
    c2 = 1.6
    max_velocity = (x_max - x_min) / 10

    score_evol = np.zeros((num_particles, num_iterations))
    global_best_history = []

    patience = 10
    min_delta = 1e-6
    var_window = 10
    var_threshold = 1e-4
    min_iterations = num_iterations // 3

    for j in tqdm(range(num_iterations), desc="Optimizing", disable=no_progress_bar):
        w = w_max - (w_max - w_min) * (j / num_iterations)

        for i, particle in enumerate(particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particle) +
                             c2 * r2 * (global_best_position - particle))
            velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
            particles[i] = np.clip(particle + velocities[i], x_min, x_max)

            score = evaluate_fitness(particles[i], top_instance, tmax, alpha, max_iter)
            score_evol[i, j] = score

            if score > personal_best_scores[i]:
                personal_best_positions[i] = particles[i]
                personal_best_scores[i] = score

                if score > global_best_score:
                    global_best_position = particles[i]
                    global_best_score = score

        global_best_history.append(global_best_score)

        if early_stopping and j >= min_iterations:
            if j >= patience:
                recent_scores = global_best_history[-patience:]
                if all(abs(recent_scores[k] - recent_scores[k-1]) < min_delta for k in range(1, patience)):
                    print(f"Early stopping at iteration {j} due to no improvement.")
                    break

            if j >= var_window:
                recent_scores = global_best_history[-var_window:]
                if np.var(recent_scores) < var_threshold:
                    print(f"Early stopping at iteration {j} due to low variance.")
                    break

    return global_best_position, global_best_score, score_evol

def evaluate_fitness(x_values, top_instance, tmax, alpha, max_iter):
    spliced_instance = instance_splice(top_instance, x_values)
    total_score = run_sol_multiple(spliced_instance, x_values, tmax, 2, alpha, max_iter, True)
    return total_score

# %%
cProfile.run('pso_optimize(TOP, 30, 50, tmax, m, 0.3, 100, no_progress_bar=False, early_stopping=False)', sort='tottime')
# %%
# Test The PSO optimization function with multiple runs to get a variance and mean score
scores = []

for i in tqdm(range(15), desc="Running multiple tests"):
    total_score = 0
    best_x, best_score, score_evol = pso_optimize(TOP, 30, 50, tmax, m, early_stopping=True, no_progress_bar=True)
    split_instance = instance_splice(TOP, best_x)
    scores.append(best_score)

print(np.var(scores))
print(np.mean(scores))
 # %% 
 #Testing the PSO optimization function with different tmax values to see how it affects the score, aswell as the variance and mean score.
tmaxs = [35, 40, 45, 50]

for tmax in tmaxs:
    scores = []

    best_x, best_score, score_evol = 0,0,0
    for i in tqdm(range(15), desc=f"Running multiple tests with tmax = {tmax}"):
        #Reset the zone pairs for each iteration
        for i in range(len(x_vals)):
            for j in range(len(x_vals)):
                zone_pairs[i, j] = [0, 0]

        total_score = 0
        best_x, best_score, score_evol = pso_optimize(TOP, 30, 50, tmax, m, no_progress_bar=True)
        scores.append(best_score)
    print(f"Variance: {np.var(scores)}, Mean: {np.mean(scores)} for tmax = {tmax}")

#%%
mmer = [2,3,4,5]
sol = []
xval = [0,0,0,0]
for i in range(len(mmer)):
    m = mmer[i]
    xval[i] = initial_splice_by_score(TOP, m)
    split_instance = instance_splice(TOP, xval[i])
    sol.append(run_sol_multiple(split_instance, tmax, 10, 0.2, 1000))

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plotting each subplot
Instance_plot.TOP_instance_plot(axs[0, 0], TOP, tmax, filepath, xval[0], sol[0])
Instance_plot.TOP_instance_plot(axs[0, 1], TOP, tmax, filepath, xval[1], sol[1])
Instance_plot.TOP_instance_plot(axs[1, 0], TOP, tmax, filepath, xval[2], sol[2])
Instance_plot.TOP_instance_plot(axs[1, 1], TOP, tmax, filepath, xval[3], sol[3])

# Set the overall title
fig.suptitle('Scores from initial split by score', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust rect to make room for the suptitle
plt.show()

# %% Used for generating a plot of the scores from the optimized zoning
mmer = [2,3,4,5]
sol = []
xval = [0,0,0,0]

for i in range(len(mmer)):
    m = mmer[i]
    xval[i], best_score, score_evol = pso_optimize(TOP, 30, 50, tmax, m, 0.3, 100)
    split_instance = instance_splice(TOP, xval[i])
    sol.append(run_sol_multiple(split_instance, tmax, 10, 0.2, 1000))

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plotting each subplot
Instance_plot.TOP_instance_plot(axs[0, 0], TOP, tmax, filepath, xval[0], sol[0])
Instance_plot.TOP_instance_plot(axs[0, 1], TOP, tmax, filepath, xval[1], sol[1])
Instance_plot.TOP_instance_plot(axs[1, 0], TOP, tmax, filepath, xval[2], sol[2])
Instance_plot.TOP_instance_plot(axs[1, 1], TOP, tmax, filepath, xval[3], sol[3])

# Set the overall title
fig.suptitle('Scores from optimized zoning', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust rect to make room for the suptitle
plt.show()
# %%
