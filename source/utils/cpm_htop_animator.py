# %%
from classes.route import Route
from library.cpm_htop_solvers import grasp
from library.core.interception.euclidian_intercepter import EuclidianInterceptionRoute
from library.core.interception.intercepter import InterceptionRoute
from classes.problem_instances.cpm_htop_instances import CPM_HTOP_Instance
from typing import List
import os 
from matplotlib import pyplot as plt 
from matplotlib import animation
import numpy as np

def animate(cpm_htop_instance: CPM_HTOP_Instance, routes: List[Route], cpm_route: InterceptionRoute, number_of_frames: int = 600):
    """Creates an animation based on the CPM-HTOP instance, a set of routes and an euclidian interception route."""
    fig, _ = plt.subplots()
    t = np.linspace(0, cpm_htop_instance.t_max, number_of_frames)

    cpm_htop_instance.plot_CPM_HTOP_solution(routes, cpm_route, show=False)
    #cpm_htop_instance.plot_with_routes(routes, plot_points=True, show=False)
    #cpm_route.plot(show=False)

    cpm_scat = plt.scatter(*cpm_route.get_position(0), c="tab:purple",  s=80, zorder=5) 
    uav_positions = [route.get_position(0) for route in routes]
    xs = [pos[0] for pos in uav_positions]
    ys = [pos[1] for pos in uav_positions]
    uav_scat = plt.scatter(xs, ys, c = cpm_htop_instance._colors[:len(routes)], s=80, zorder=1)

    def update(frame):
        """Updates the figure"""
        x, y = cpm_route.get_position(t[frame])
        cpm_scat.set_offsets(np.stack([[x], [y]]).T)

        uav_positions = [route.get_position(t[frame]) for route in routes]
        xs = [pos[0] for pos in uav_positions]
        ys = [pos[1] for pos in uav_positions]
        uav_scat.set_offsets(np.stack([xs, ys]).T)

        plt.title(f"Instance: {cpm_htop_instance.problem_id}, time: {round(t[frame], 2)} / {cpm_htop_instance.t_max}")

        return cpm_scat, uav_scat

    _ = animation.FuncAnimation(fig=fig, func=update, frames=number_of_frames, interval=1)
    plt.show()

if __name__ == "__main__":
    folder_with_top_instances = os.path.join(os.getcwd(), "resources", "CPM_HTOP") 
    problem_id = "p4.4.f.2"
    cpm_htop_instance = CPM_HTOP_Instance.load_from_file(f"{problem_id}.txt", needs_plotting=True)
    print(f"Running GRASP on {problem_id}")
    routes = grasp(cpm_htop_instance, p = 0.7, time_budget = 10, use_centroids = True)
    euclidian_cpm_route = EuclidianInterceptionRoute([0, 2, 3, 1], routes, cpm_htop_instance.cpm_speed, 1, cpm_htop_instance.source.pos, cpm_htop_instance.sink.pos)
    
    #EuclidianInterceptionRoute(cpm_htop_instance.source.pos, [0, 2, 3, 1], routes, cpm_htop_instance.cpm_speed, cpm_htop_instance.sink.pos, delay=1)
    animate(cpm_htop_instance, routes, euclidian_cpm_route)