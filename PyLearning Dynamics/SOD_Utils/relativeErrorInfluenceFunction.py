import time
from pytictoc import TicToc
from SOD_Utils.L2_rho_T_energy import L2_rho_T_energy
from SOD_Utils. L2_rho_T_alignment import  L2_rho_T_alignment
from SOD_Utils.L2_rho_T_xi import L2_rho_T_xi
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_prey import PS_1st_order_prey_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_prey import PS_1st_order_predator_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_predator import PS_1st_order_prey_on_predator
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_predator import PS_1st_order_predator_on_predator

def relativeErrorInfluenceFunction(phihat, phi, sys_info, obs_info, basis, type):
    """
    Calculate the relative and absolute error of influence functions.

    Parameters:
        phihat (dict): The estimated influence functions.
        phi (dict): The true influence functions.
        sys_info (dict): Information about the dynamical system.
        obs_info (dict): Information about observations.
        basis (dict): Basis functions.
        type (str): Type of the influence function.

    Returns:
        Err (dict): Dictionary containing absolute and relative errors for each influence function.
    """
    flag = False
    if isinstance(phihat,dict):
       if len(phihat.keys()) > 1:
           flag = True
    else:
        if phihat.size == 4:
            flag = True
    
    
    phidiff = {}
    for k_1 in range(sys_info['K']-1, -1, -1):
        for k_2 in range(sys_info['K']-1, -1, -1):
            #phidiff[k_1, k_2] = phi[k_1, k_2] - phihat[k_1, k_2]
            fun = {}
            if flag :
                fun = {'PS': phi[k_1, k_2] ,'lc': phihat[k_1, k_2] }
                phidiff[k_1, k_2] = fun
            else:
                fun = {'PS': phi[k_1, k_2] }
                phidiff[k_1, k_2] = fun
           

    Err = {}
    Err['Timings'] = {'L2rhoT': 0}

    t = TicToc()
    t.tic()

    switcher = {
        'energy': L2_rho_T_energy,
        'alignment': L2_rho_T_alignment,
        'xi': L2_rho_T_xi
    }
    L2_rho_T = switcher.get(type, lambda a, b, c, d: None)

    phi_L2norms = L2_rho_T(phi, sys_info, obs_info, basis)
    phihat_L2rhoTdiff = L2_rho_T(phidiff, sys_info, obs_info, basis)

    Err['Timings']['L2rhoT'] = t.tocvalue()

    Err['Abs'] = {}
    Err['Rel'] = {}

    for k_1 in range(sys_info['K'] ):
        for k_2 in range(sys_info['K'] ):
            Err['Abs'][k_1, k_2] = phihat_L2rhoTdiff[k_1][k_2]
            if phi_L2norms[k_1][k_2] != 0:
                Err['Rel'][k_1, k_2] = phihat_L2rhoTdiff[k_1][k_2] / phi_L2norms[k_1][k_2]
            else:
                if phihat_L2rhoTdiff[k_1][k_2] == 0:
                    Err['Rel'][k_1, k_2] = 0
                else:
                    Err['Rel'][k_1, k_2] = float('inf')

    return Err
