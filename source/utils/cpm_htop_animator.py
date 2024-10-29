from classes.route import Route
from library.cpm_htop_solvers import grasp
from core.euclid_interception import EuclidianInterceptionRoute
from classes.problem_instances.cpm_htop_instances import CPM_HTOP_Instance
from typing import List
import os 
from matplotlib import pyplot as plt 
from matplotlib import animation
import numpy as np

def animate(cpm_htop_instance: CPM_HTOP_Instance, routes: List[Route], euclidian_interception_route: EuclidianInterceptionRoute):
    """Creates an animation based on the CPM-HTOP instance, a set of routes and an euclidian interception route."""
    fig, _ = plt.subplots()
    t = np.linspace(0, cpm_htop_instance.t_max, 300)
    print(cpm_htop_instance.t_max)

    cpm_htop_instance.plot_with_routes(routes, plot_points=True, show=False)
    euclidian_interception_route.plot(show=False)

    cpm_scat = plt.scatter(*euclidian_interception_route.get_position_at_time(0), c="tab:red") 
    uav_positions = [route.get_position_at_time(0) for route in routes]
    xs = [pos[0] for pos in uav_positions]
    ys = [pos[1] for pos in uav_positions]
    uav_scat = plt.scatter(xs, ys, c = cpm_htop_instance._colors[:len(routes)])

    def update(frame):
        """Updates the figure"""
        x, y = euclidian_interception_route.get_position_at_time(t[frame])
        cpm_scat.set_offsets(np.stack([[x], [y]]).T)

        uav_positions = [route.get_position_at_time(t[frame]) for route in routes]
        xs = [pos[0] for pos in uav_positions]
        ys = [pos[1] for pos in uav_positions]
        uav_scat.set_offsets(np.stack([xs, ys]).T)

        return cpm_scat, uav_scat

    _ = animation.FuncAnimation(fig=fig, func=update, frames=300, interval=30)
    plt.show()
    


if __name__ == "__main__":
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "CPM_HTOP") 
    cpm_htop_instance = CPM_HTOP_Instance.load_from_file("p4.3.f.0.txt", needs_plotting=True)
    print("Running Algorithm")
    routes = grasp(cpm_htop_instance, p = 0.7, time_budget = 10, use_centroids = True)
    print("Finished Algorithm")
    euclidian_interception_route = EuclidianInterceptionRoute(cpm_htop_instance.source.pos, [0, 2, 1, 0, 1, 2, 0], routes, 1.6, cpm_htop_instance.sink.pos, waiting_time=1)
    animate(cpm_htop_instance, routes, euclidian_interception_route)