"""This module contains the code for Boustrophedon coverage path planning."""

import gurobipy as gp 
from gurobipy import GRB
from shapely import Polygon, MultiPolygon, Point, LineString
import shapely
from classes.data_types import Angle, Position
import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Iterable

def swath_generator_for_convex_area(area: Polygon, number_of_angles_to_attempt: int, sweep_width: float) -> List[LineString]:
    """Generates a set of swaths, by minimizing the number of swaths generated, which can be joined according to a path generator."""
    assert sweep_width > 0, "We cannot have a negative sweep width"

    number_of_swaths_if_theta_is_chosen, theta_of_rotated_swaths = np.inf, None
    for theta in np.pi * np.linspace(0, number_of_angles_to_attempt, number_of_angles_to_attempt, endpoint = False) / number_of_angles_to_attempt:
        rotated_area = shapely.affinity.rotate(area, theta, use_radians = True, origin=area.centroid)
        # Compute number of swaths from the centroid of the area.
        xs_of_rotated, _ = rotated_area.exterior.xy
        x_min, x_max = min(xs_of_rotated), max(xs_of_rotated)
        if (number_of_swaths_if_theta_is_chosen > (number_of_swaths := np.ceil(abs(x_min - x_max - sweep_width) / sweep_width))):
            number_of_swaths_if_theta_is_chosen = int(number_of_swaths)
            theta_of_rotated_swaths = theta

    # Actually generate the swaths.
    rotated_area = shapely.affinity.rotate(area, theta_of_rotated_swaths, use_radians = True, origin=area.centroid)
    xs_of_rotated, ys_of_rotated = rotated_area.exterior.xy
    x_min, x_max = min(xs_of_rotated), max(xs_of_rotated)
    y_min, y_max = min(ys_of_rotated), max(ys_of_rotated)
    
    unclipped_swaths = [LineString([(x, y_min), (x, y_max)]) for x in np.linspace(x_min + sweep_width / 2, x_max - sweep_width / 2, number_of_swaths_if_theta_is_chosen)]
    clipped_swaths = [rotated_area.intersection(swath) for swath in unclipped_swaths]
    
    # Revert to the original coordinate system.
    return [shapely.affinity.rotate(swath, -theta_of_rotated_swaths, use_radians = True, origin=area.centroid) for swath in clipped_swaths]
    
def compute_distance(fro: Position, to: Position, obstacles: Iterable[Polygon]) -> float:
    """Computes the minimum distance between the fro and to positions, subject to the polygons"""
    return np.linalg.norm(fro - to) # TODO

def generate_coverage_path_for_convex_areas(areas: Iterable[Polygon], obstacles: Iterable[Polygon], initial_position: Position, terminal_position: Position, number_of_angles_to_attempt: Angle, sweep_width: float, time_budget: float = 30.0) -> List[LineString]:
    """Generates a coverage path plan over all polygons with the given parameters"""
    # 1. Generate swaths for each polygonal area
    swaths_for_area = [swath_generator_for_convex_area(area, number_of_angles_to_attempt, sweep_width) for area in areas]

    # 2. Compute entry / exit points of each of the areas and compute distance between each pair
    initial_points = {area_idx: np.array([swaths[0].coords[0], swaths[0].coords[1], swaths[-1].coords[0], swaths[-1].coords[1]]) for area_idx, swaths in enumerate(swaths_for_area)}
    terminal_points = {area_idx: np.array([swaths[-1].coords[0], swaths[-1].coords[1], swaths[0].coords[0], swaths[0].coords[1]]) if (len(swaths) % 2) == 0 else 
                       np.array([swaths[-1].coords[1], swaths[-1].coords[0], swaths[0].coords[1], swaths[0].coords[0]]) for area_idx, swaths in enumerate(swaths_for_area)}
    M = len(areas)
    N = 4 * M + 2

    d = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != 0 and i != N - 1:
                frm_i, to_i = terminal_points[((i - 1) // 4)][(i - 1) % 4], initial_points[((i - 1) // 4)][(i - 1) % 4] 
            elif i == 0:
                frm_i, to_i = initial_position, initial_position
            else:
                frm_i, to_i = terminal_position, terminal_position

            if j != 0 and j != N - 1:
                frm_j, to_j = terminal_points[((j - 1) // 4)][(j - 1) % 4], initial_points[((j - 1) // 4)][(j - 1) % 4] 
            elif j == 0:
                frm_j, to_j = initial_position, initial_position
            else:
                frm_j, to_j = terminal_position, terminal_position

            d[i, j] = compute_distance(frm_i, to_j, obstacles)
            d[j, i] = compute_distance(frm_j, to_i, obstacles)
    
    print(d)
        
    # 3. Compute Generalized Path TSP over polygonal areas TODO:

    # 3.1 Setup a gurobi model for computing the path TSP
    model = gp.Model("Generalized path-TSP model for area ordering & route specification")
    x = model.addVars(range(N), range(N), vtype=GRB.BINARY, name="x")
    u = model.addVars(range(N), vtype=GRB.INTEGER, name="u", lb = 2, ub = N)

    ## 3.2 Setup objective and constraint
    model.setObjective(gp.quicksum(d[i, j] * x[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE) 

    model.addConstr(gp.quicksum(x[0, i] for i in range(1, N)) == 1, name = "Departs source")
    #model.addConstr(gp.quicksum(x[i, 0] for i in range(1, N)) == 0, name = "Never Arrives at the source")
    model.addConstr(gp.quicksum(x[i, N - 1] for i in range(N - 1)) == 1, name = "Arrives at depot")
    #model.addConstr(gp.quicksum(x[N - 1, i] for i in range(N - 1)) == 0, name = "Never Departs the depot")

    model.addConstrs((gp.quicksum(x[i, j] for i in range(4 * k + 1, 4 * k + 5) for j in range(N) if (j not in set(range(4 * k + 1, 4 * k + 5)))) == 1 for k in range(M)), name="Departs from each area once")
    model.addConstrs((gp.quicksum(x[j, i] for i in range(4 * k + 1, 4 * k + 5) for j in range(N) if (j not in set(range(4 * k + 1, 4 * k + 5)))) == 1 for k in range(M)), name="Arrives at each area once")

    model.addConstrs((gp.quicksum(x[i, j] - x[j, i] for j in range(N) if j != i) == 0 for i in range(1, N - 1)), name="Flow conservation constraints")
    model.addConstrs((u[i] - u[j] + 1 <= (N - 1) * (1 - x[i, j]) for i in range(1, N) for j in range(1, N)), name="Subtour Elimination Constraints")
    #model.addConstr((u[p] - u[q] + (N - 1) * gp.quicksum() + (N - 3) * gp.quicksum() <= (N - 1)) for p in range(M) for q in range(M) if p != q)

    model.setParam("TimeLimit", time_budget)
    model.optimize()
    
    # Convert MILP output to actual sweep path 
    used = set()
    for i in range(N):
        for j in range(N):
            if x[i, j].x == 1.0:
                used.add(i)
                used.add(j)

                #fro = (terminal_points[((i - 1) // 4)][(i - 1) % 4] if i != 0 and i != N - 1 else (initial_position if i == 0 else terminal_position))
                #to = (initial_points[((j - 1) // 4)][(j - 1) % 4] if j != 0 and j != N - 1 else (initial_position if j == 0 else terminal_position))
                #xs = [fro[0], to[0]]
                #ys = [fro[1], to[1]]
                #print(np.linalg.norm(fro - to))
                #plt.scatter(xs, ys, color="tab:green", zorder=3)
                #plt.plot(xs, ys, color="tab:green", zorder=3)

    area_ordering = sorted(list(used.difference({0, N - 1})), key = lambda i: u[i].x)
    #print(area_ordering, N)
    
    # Generate route based on the ordering of the polygonal areas
    # For now simply pick the starting vertex greedily within each area, i.e. the swath closest to the current position
    sweep_path = []
    current_position = initial_position
    for idx in area_ordering:
        area_idx = (idx - 1) // 4
        swaths = swaths_for_area[area_idx]

        swath_starting_positions = list(map(np.array, [swaths[0].coords[0], swaths[0].coords[1], swaths[-1].coords[0], swaths[-1].coords[1]]))

        starting_point_idx = (idx - 1) % 4
        sweep_path.append(LineString([current_position, swath_starting_positions[starting_point_idx]]))

        # Generate the route based on the swaths:
        flip = (starting_point_idx % 2 == 1)
        for swath in (reversed(swaths) if starting_point_idx >= 2 else swaths):
            sweep_path.append(LineString([current_position, swath.coords[1 if flip else 0]]))
            sweep_path.append(LineString([swath.coords[1 if flip else 0], swath.coords[0 if flip else 1]]))
            current_position = swath.coords[0 if flip else 1]
            flip = not flip

    sweep_path.append(LineString([current_position, terminal_position]))

# Arman, meld tilbage ift til seminar.

    return sweep_path

if __name__ == "__main__":
    areas = [Polygon([(0.0, 0.0), (0.2, 1.0), (0.3, 1.1), (0.8, 1.2), (0.6, 0.0)]), Polygon([(2, 2), (3, 2), (2.5, 3)])]
    for area in areas:
        x, y = area.exterior.xy
        plt.plot(x, y, color = "black", linestyle="dashed", zorder=1)

    initial_position = np.array([-1, -1]) 
    terminal_position = np.array([4, 4])
    plt.scatter(initial_position[0], initial_position[1], color="tab:orange", zorder = 3)
    plt.scatter(terminal_position[0], terminal_position[1], color="tab:green", zorder = 3)

    sweep = generate_coverage_path_for_convex_areas(areas, [], initial_position, terminal_position, 10, 0.1)
    for sweep_segment in sweep:
        x, y = sweep_segment.xy
        plt.plot(x, y, color = "tab:blue", zorder=2)

    plt.show()
        
    #swaths = swath_generator_for_convex_area(area, number_of_angles_to_attempt = 20, sweep_width = 0.1)
    #for swath in swaths:
    #    x, y = swath.xy
    #    plt.plot(x, y, color="tab:blue")

    #x, y = area.exterior.xy
    #plt.grid(True)
    #plt.plot(x, y, label="Polygon", color="black")
    #plt.show()