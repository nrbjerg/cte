from classes.problem_instances.cedopads_instances import CEDOPADSNode, Visit
import numpy as np 
from typing import Tuple, Iterable

def equidistant_sampling(node: CEDOPADSNode, eta: float, sensing_radii: Tuple[float, float]) -> Iterable[Visit]:
    """Simply samples the visits at the middle of each AOA""" 
    r = sum(sensing_radii) / 2
    if eta <= np.pi / 2:
        tau_offsets = [np.pi, np.pi + eta / 2, np.pi - eta / 2]
    else:
        tau_offsets = [np.pi, np.pi + eta / 3, np.pi - eta / 3, np.pi + 2 * eta / 3, np.pi - 2 * eta / 3]

    for theta, phi in zip(node.thetas, node.phis):
        if phi <= np.pi / 2:
            psis = [theta, (theta + phi / 2) % (2 * np.pi), (theta - phi / 2) % (2 * np.pi)]
        else: #phi <= 2 * np.pi / 3:
            psis = [theta, (theta + phi / 3) % (2 * np.pi), (theta - phi / 3) % (2 * np.pi),
                           (theta + 2 * phi / 3) % (2 * np.pi), (theta - 2 * phi / 3) % (2 * np.pi)]
    
        for psi in psis:
            for tau_offset in tau_offsets:
                yield (node.node_id, psi, (psi + tau_offset) % (2 * np.pi), r)