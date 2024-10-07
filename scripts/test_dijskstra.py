from classes.problem_instances.top_instances import TOPInstance, load_TOP_instances
from core.dijskstra import reversed_dijkstra

initial_top_instance = load_TOP_instances(needs_plotting=True)[0]
routes = reversed_dijkstra(initial_top_instance.nodes)
initial_top_instance.plot_with_routes([routes[0]])