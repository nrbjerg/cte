# %%
from typing import List
from classes.route import Route
from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
import json
import os 

def save_list_of_routes_as_json_file(routes: List[Route], file_path: str):
    """Saves the routes in a file at the given file path."""
    simplified_routes = [route.get_node_ids() for route in routes]
    with open(file_path, "w+") as json_file:
        json.dump(simplified_routes, json_file)

def load_list_of_routes_from_json_file(file_path: str, problem_instance: TOPInstance) -> List[Route]:
    """Loads the routes stored in the file specified via the file path."""
    with open(file_path, 'r') as json_file:
        simplied_routes = json.load(json_file)

        return [Route([problem_instance.nodes[node_id] for node_id in simplified_route]) for simplified_route in simplied_routes]

if __name__  == "__main__":
    problem_instance = load_TOP_instances(needs_plotting = True, neighbourhood_level = 1)[0]
    file_path = os.path.join(os.getcwd(), "test.json")
    save_list_of_routes_as_json_file([Route([problem_instance.source, problem_instance.sink])], file_path)
    print(load_list_of_routes_from_json_file(file_path, problem_instance))
