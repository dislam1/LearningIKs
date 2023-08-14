import numpy as np
from SOD_Evolve.find_pair_diff import find_pair_diff
from SOD_Evolve.find_pair_angle import find_pair_angle
from SOD_Evolve.find_phis_of_pdist import find_phis_of_pdist
from SOD_Evolve.add_regulation import add_regulation


def find_collective_change(x, v, xi, pdist_mat, sys_info, type):
    """
    Calculates the collective change: sum_{i' \in K_i'} kappa_{K_i'}/N_i' * \phi_{K_i, K_i'}(|x_i - x_i'|) * pdiff_ii'
    
    Args:
    x: x_i's, state of agents (position, opinions, etc.)
    v: dot{x}_i's, derivative of the state
    xi: xi_i's, interaction with the environment or emotion (fear, etc.)
    pdist_mat: pairwise distance matrix (|x_i - x_i'|)
    sys_info: dict containing necessary information of the system
    type: str, type of the regulation (energy, alignment, or xi based)

    Returns:
    collective_change: sum_{i' \in K_i'} kappa_{K_i'}/N_i' * \phi_{K_i, K_i'}(|x_i - x_i'|) * pdiff_ii'

    """
    # Prepare phis_of_pdist (with regulation)
    if 'has_theta' in sys_info and sys_info['has_theta']:
        theta_mat = find_pair_angle(x, v)
    else:
        theta_mat = None
    
    phis_of_pdist = find_phis_of_pdist(pdist_mat, theta_mat, sys_info, type)
    
    phis_of_pdist = add_regulation(x, v, xi, phis_of_pdist, sys_info, type)

    # Prepare pairwise differences
    if type == 'energy':
        the_pdiff = find_pair_diff(x)
        if sys_info['d'] > 1:
            phis_of_pdist = np.kron(phis_of_pdist, np.ones((sys_info['d'], 1)))
    elif type == 'alignment':
        the_pdiff = find_pair_diff(v)
        if sys_info['d'] > 1:
            phis_of_pdist = np.kron(phis_of_pdist, np.ones((sys_info['d'], 1)))
    elif type == 'xi':
        the_pdiff = find_pair_diff(xi)
    else:
        return
    
    # Pointwise multiplication of kappa_{K_i'}/N_i' * \phi_{K_i, K_i'}(|x_i - x_i'|) * pdiff_ii'
    agent_change = phis_of_pdist * the_pdiff
    collective_change = np.sum(agent_change, axis=1)
    
    return collective_change
