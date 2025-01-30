# %%
import scipy
from classes.data_types import Dir, State 
from typing import Optional, Tuple, List
from library.core.dubins.dubins_api import call_dubins, plot_path
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def elongate(q_i: State, q_f: State, rho: float, d: float) -> Optional[Tuple[List[Dir], List[float]]]:
    """Elongates the shortest dubins path between q_i and q_f to a length of d if posible."""
    (shortest_typ, shortest_seg_lengths) = call_dubins(q_i, q_f, rho)
    length_of_shortest_path = sum(shortest_seg_lengths)

    assert d > length_of_shortest_path

    delta = d - length_of_shortest_path
    if shortest_typ == (Dir.L, Dir.R, Dir.L) or shortest_typ == (Dir.R, Dir.L, Dir.R):
        # We can use Lemma 1 of "elongation of curvature-bounded path"
        return ([shortest_typ[0], Dir.S, shortest_typ[1], Dir.S, shortest_typ[1], shortest_typ[2]],
                [shortest_seg_lengths[0], delta / 2, rho * np.pi, delta / 2, shortest_seg_lengths[1] - rho * np.pi, shortest_seg_lengths[2]])
    
    else: # We are either in LSL, RSR, LSR or RSL
        if shortest_seg_lengths[1] > 4 * rho:
            # We can use Lemma 2 of "elongation of curvature-bounded path"
            if delta > (2 * np.pi * rho - 4 * rho):
                alpha = (shortest_seg_lengths[1] - 4 * rho) / 2 # The length before and after the kinks
                beta = (delta - (2 * np.pi * rho - 4 * rho)) / 2 # The length of the straight line segments
                print(f"{beta=}")
                return (
                    [shortest_typ[0], Dir.S, shortest_typ[0], Dir.S, shortest_typ[0].flip(), Dir.S, shortest_typ[0], Dir.S, shortest_typ[2]],
                    [shortest_seg_lengths[0], alpha, np.pi / 2 * rho, beta, np.pi * rho, beta, np.pi / 2 * rho, alpha, shortest_seg_lengths[2]]
                )
            
            else:
                theta = scipy.optimize.newton(lambda theta: (2 * rho + delta / 2) - 2 * rho * (theta + (1 - np.sin(theta)) / np.cos(theta)), np.pi / 2) #  (0, np.pi / 2))
                print(f"{(2 * rho + delta / 2) - 2 * rho * (theta + (1 - np.sin(theta)) / np.cos(theta))=}")
                alpha = (shortest_seg_lengths[1] - 4 * rho) / 2 # Length of before and after the kinks
                beta =  2 * rho * (1 - np.sin(theta)) / np.cos(theta) # The length of the staright line segments after the first kink and before the last kink
                print(f"{beta=}, {theta=}")
                return (
                    [shortest_typ[0], Dir.S, shortest_typ[0], Dir.S, shortest_typ[0].flip(), Dir.S, shortest_typ[0], Dir.S, shortest_typ[2]],
                    [shortest_seg_lengths[0], alpha, theta * rho, beta, 2 * theta * rho, beta, theta * rho, alpha, shortest_seg_lengths[2]]
                )

        elif shortest_seg_lengths[0] >= np.pi * rho:
            # Use elongation strategy from figure 6.a
            beta = delta / 2 # Lengths of the inserted line segments
            return (
                [Dir.S, shortest_typ[0], Dir.S, shortest_typ[0], Dir.S, shortest_typ[2]],
                [beta, np.pi * rho, beta, shortest_seg_lengths[0] - np.pi * rho, shortest_seg_lengths[1], shortest_seg_lengths[2]]
            )

        elif shortest_seg_lengths[2] >= np.pi * rho:
            beta = delta / 2 # Lengths of the inserted line segments
            return (
                [shortest_typ[0], Dir.S, shortest_typ[2], Dir.S, shortest_typ[2], Dir.S],
                [shortest_seg_lengths[0], shortest_seg_lengths[1], shortest_seg_lengths[2] - np.pi * rho, beta, np.pi * rho, beta]
            )           # Use elongation strategy from figure 6.b
        else:
            # TODO
            pass 

        
        
if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = [9, 9]
    q_i = State(np.array([0, 0]), np.pi / 2)
    q_f = State(np.array([6, -6]), np.pi / 2)
    rho = 2
    (typ, seg_lengths) = elongate(q_i, q_f, rho, 20)
    print(f"{typ=}{seg_lengths=}{sum(seg_lengths)=}")
    plot_path(q_i, typ, seg_lengths, rho, color = "tab:orange")
    (shortest_typ, shortest_seg_lengths) = call_dubins(q_i, q_f, rho)
    print(f"{sum(shortest_seg_lengths)=}")
    plot_path(q_i, shortest_typ, shortest_seg_lengths, rho, linestyle=":")
