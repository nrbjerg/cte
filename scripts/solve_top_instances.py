# This scripts solves all of the TOP problems using the GRASP algorithm
import os

from classes.problem_instances.top_instances import load_TOP_instances
from library.top_solvers import grasp
from tqdm import tqdm
from utils.route_archiving import save_list_of_routes_as_json_file

folder_with_top_solutions = os.path.join(os.getcwd(), "archive", "TOP_solutions")
already_solved_top_instances = [".".join(file.split(".")[:-1]) for file in os.listdir(folder_with_top_solutions)]
top_instances = list(filter(lambda top_instance: top_instance.problem_id not in already_solved_top_instances, load_TOP_instances()))
for top_instance in tqdm(top_instances):
    routes = grasp(top_instance, 60, 0.7, use_centroids=True)
    save_list_of_routes_as_json_file(routes, os.path.join(folder_with_top_solutions, f"{top_instance.problem_id}.json"))
