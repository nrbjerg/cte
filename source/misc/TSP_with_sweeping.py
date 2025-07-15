from typing import List, Iterable, Tuple
import matplotlib.pyplot as plt 

Position = Tuple[int, int]

def sweep(points: Iterable[Position], x_min: int, x_max: int, y_goal: int, distance_budget: float) -> List[Position]:
    """Computes a sweeping patern of a given distance, within the square {x_min, ..., x_max} x {0, ..., y_goal}. 
       For the sake of simplicity we assume that there is no targets on the first row."""
    width = x_max - x_min
    rows = dict()
    for point in points:
        rows[point[1]] = rows.get(point[1], []) + [point[0]]
    
    ys = sorted(point[1] for point in points)

    sweep_from_left = True
    distance_sweept = abs(x_min)
    route = [(0, 0), (x_min, 0)]
    last_row_sweept = 0
    for row_idx in range(1, y_goal + 1):
        # Remaining distance budget uppon reaching the point route[-1] from (0, 0)
        remaining_distance_budget = distance_budget - (distance_sweept + (row_idx - last_row_sweept)) 
        minimum_remaning_distance = width * len({y for y in ys if y >= row_idx}) + (y_goal - row_idx)
        if row_idx in ys or minimum_remaning_distance + width < remaining_distance_budget:
            # Sweep the line 
            xs = sorted(rows.get(row_idx, []), reverse=sweep_from_left)
            route.append((x_min, row_idx) if sweep_from_left else (x_max, row_idx))
            route.extend([(x, row_idx) for x in xs])
            route.append((x_max, row_idx) if sweep_from_left else (x_min, row_idx))
            sweep_from_left = not sweep_from_left
            distance_sweept += row_idx - last_row_sweept + width
            last_row_sweept = row_idx
    
    if route[-1][1] != y_goal:
        route.append((x_max, y_goal) if not sweep_from_left else (x_min, y_goal))

    return route

def add_orientations_to_route(route: List[Position]) -> str:
    """Covnerts the route list to a list of waypoints."""
    # Setup theta values.
    east = 0.0
    west = 180.0
    north = 90.0
    south = 270.0

    waypoints = [f"-{{x: {float(route[0][0])}, y: {float(route[0][1])}, theta: {east}}}"]
    for p, q in zip(route[:-1], route[1:]):
        # Compute the orientation for the UXV moving from waypoint p to waypoint q.
        if p[0] < q[0]:
            waypoints.append(f"-{{x: {float(q[0])}, y: {float(q[1])}, theta: {south}}}")
        elif p[0] > q[0]:
            waypoints.append(f"-{{x: {float(q[0])}, y: {float(q[1])}, theta: {north}}}")
        elif p[1] > q[1]: 
            waypoints.append(f"-{{x: {float(q[0])}, y: {float(q[1])}, theta: {west}}}")
        else:
            # NOTE: this should never be used, since we are always moving forward!
            waypoints.append(f"-{{x: {float(q[0])}, y: {float(q[1])}, theta: {east}}}")

    return "waypoints_sweep:\n" + "\n".join(waypoints)

def add_intermediate_waypoints(route: List[Position], max_distance: int = 3) -> List[Position]:
    """Adds extra waypoints to the route to make sure that the max distance is at most max_distance"""
    route_with_intermediate_waypoints = [route[0]]
    for p, q in zip(route[:-1], route[1:]):
        distance_between_p_and_q = abs(p[0] - q[0]) + abs(p[1] - q[1])
        if distance_between_p_and_q > max_distance:
            for k in range(1, distance_between_p_and_q // max_distance):
                print(k)
                if p[0] < q[0]:
                    route_with_intermediate_waypoints.append((p[0] + k * max_distance, p[1]))
                elif p[0] > q[0]:
                    route_with_intermediate_waypoints.append((p[0] - k * max_distance, p[1]))
                elif p[1] < q[1]:
                    route_with_intermediate_waypoints.append((p[0], p[1] + k * max_distance))
                else:
                    route_with_intermediate_waypoints.append((p[0], p[1] - k * max_distance))

        route_with_intermediate_waypoints.append(q)

    return route_with_intermediate_waypoints

if __name__ == "__main__":
    points = [(1, 2),
              (4, 4),
              (-2, 1),
              (-3, 2),
              (-1, 6)]
    
    distance_budget = 50
    
    
    # Plot route initial route
    plt.style.use("ggplot")
    plt.scatter([p[0] for p in points], [p[1] for p in points], c = "tab:blue", linestyle="-", zorder=2)
    
    # Plot route with sweeping
    sweeping_route = sweep(points, x_min = -3, x_max = 5, y_goal = 6, distance_budget = distance_budget)
    plt.plot([p[0] for p in sweeping_route], [p[1] for p in sweeping_route], c = "tab:orange", linestyle="-")

    # Add waypoints
    sweeping_route_with_intermediate_waypoints = add_intermediate_waypoints(sweeping_route)
    print(sweeping_route_with_intermediate_waypoints)
    plt.scatter([p[0] for p in sweeping_route_with_intermediate_waypoints], [p[1] for p in sweeping_route_with_intermediate_waypoints], c="tab:orange")

    plt.title(f"Total distance budget: {distance_budget}")
    plt.show()

    # Convert sweep to YAML format.
    yaml_route = add_orientations_to_route(sweeping_route)
    with open("sweep.yml", "w+") as file:
        file.write(yaml_route)