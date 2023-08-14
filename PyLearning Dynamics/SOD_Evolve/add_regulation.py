# -*- coding: utf-8 -*-
def add_regulation(x, v, xi, phis_of_pdist, sys_info, type):
    """
    Parameters
    ----------
    x : Array
        x_i's, state of agents (position, opinions, etc.)
    v : Array
        derivative of the state.
    xi : Array
        xi_i's, interaction with the environment or emotion (fear, etc.).
    phis_of_pdist : Array
        phi_{k1, k2}(|x_i - x_i'|).
    sys_info : Object
        struct which contains necessary information of the system.
    type : STR
        type of the regulation (energy, alignment or xi based).

    Returns
    -------
    phis_of_pdist : Array
        modified phis_of_pdist.

    """
    
    
    # Find out the regulation to constrain the dynamics based on type
    if type == 'energy':
        if sys_info['RE'] is not None:
            regulation = sys_info['RE'](x)  # Energy-based regulation depends only on x
        else:
            regulation = []
    elif type == 'alignment':
        if sys_info['RA'] is not None:
            regulation = sys_info['RA'](x, v)  # Alignment-based regulation depends on x and v
        else:
            regulation = []
    elif type == 'xi':
        if sys_info['Rxi'] is not None:
            regulation = sys_info['Rxi'](x, xi)  # Xi-based regulation depends on x and xi
        else:
            regulation = []
    else:
        regulation = []
    
    # Modify each phi_of_pdist if there is any regulation
    if regulation:
        phis_of_pdist = phis_of_pdist * regulation
    
    return phis_of_pdist


