import random 
from scipy.spatial.distance import cdist
import import_TOP_instance
import numpy as np
import time
filepath = "C:/Users/NX83SQ/GitHub/Benchmark_instances/Set_100/p4.4.m.txt"
n, m, tmax = import_TOP_instance.import_numbers(filepath)[0], import_TOP_instance.import_numbers(filepath)[1], import_TOP_instance.import_numbers(filepath)[2]
temp_TOP = import_TOP_instance.parse_top_file(filepath)
TOP = temp_TOP[1:-1]

def compute_distances(node_data):
    coordinates = node_data[:, :2]
    return cdist(coordinates, coordinates, metric='euclidean')

def compute_distances_and_ratios(node_data):
    scores = node_data[:, 2]
    distances = compute_distances(node_data)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_matrix = np.where(distances != 0, scores[None, :] / distances, 0)

    combined = np.stack((distances, ratio_matrix), axis=2)  # Shape: (n, n, 2)
    return combined

def GRASP_OP_optimized(TOP, tmax, alpha=0.3, max_iter=100):
    n = TOP.shape[0]
    scores = TOP[:, 2]
    distances = compute_distances_and_ratios(TOP)
    sink = n - 1

    def greedy_random_construct():
        solution = [0]
        time_spent = 0
        available_nodes = np.arange(1, n - 1)

        while available_nodes.size > 0:
            last_node = solution[-1]

            # Extract travel and return times from the distance matrix (channel 0)
            travel_times = distances[last_node, available_nodes, 0]
            return_times = distances[available_nodes, sink, 0]
            total_times = time_spent + travel_times + return_times

            feasible_mask = total_times < tmax
            feasible_nodes = available_nodes[feasible_mask]

            if feasible_nodes.size == 0:
                break

            # ✅ Use precomputed score-to-distance ratios (channel 1)
            ratios = distances[last_node, feasible_nodes, 1]

            max_ratio = np.max(ratios)
            threshold = max_ratio - alpha * max_ratio
            rcl_mask = ratios >= threshold
            rcl = feasible_nodes[rcl_mask]

            next_node = np.random.choice(rcl)
            solution.append(next_node)
            time_spent += distances[solution[-2], next_node, 0]
            available_nodes = available_nodes[available_nodes != next_node]

        solution.append(sink)
        return solution
    
    def is_feasible(solution):
        path = distances[solution[:-1], solution[1:],0]
        return np.sum(path) <= tmax


    def local_search(solution):
        best_solution = solution[:]
        best_score = np.sum(scores[best_solution])

        for _ in range(5):  # limit number of local search iterations
            if len(best_solution) <= 2:
                break

            # Randomly remove a node (not start or end)
            remove_idx = np.random.randint(1, len(best_solution) - 1)
            candidate_solution = best_solution[:remove_idx] + best_solution[remove_idx + 1:]

            # Find nodes not in the current solution
            current_nodes = set(candidate_solution)
            available_nodes = np.setdiff1d(np.arange(1, n - 1), list(current_nodes))

            if available_nodes.size == 0:
                continue

            # Use precomputed score-to-distance ratios
            last_node = candidate_solution[-2]
            ratios = distances[last_node, available_nodes, 1]
            sorted_indices = np.argsort(-ratios)
            sorted_nodes = available_nodes[sorted_indices]

            for node in sorted_nodes:
                new_solution = candidate_solution[:-1] + [node] + [sink]
                if is_feasible(new_solution):
                    new_score = np.sum(scores[new_solution])
                    if new_score > best_score:
                        best_solution = new_solution
                        best_score = new_score
                    break

        return best_solution

    best_score = 0
    best_solution = []

    for _ in range(max_iter):
        initial_solution = greedy_random_construct()
        improved_solution = local_search(initial_solution)
        current_score = np.sum(scores[improved_solution])

        if current_score > best_score:
            best_solution = improved_solution
            best_score = current_score

    route_points = TOP[best_solution]
    return route_points


if __name__ == "__main__":
    
    # dist = compute_distances_and_ratios(TOP)
    # print("Distances shape:", dist.shape)
    
    
    # print(compute_distances(TOP))

    solution = GRASP_OP_optimized(TOP, tmax)

    # print(np.shape(solution))
