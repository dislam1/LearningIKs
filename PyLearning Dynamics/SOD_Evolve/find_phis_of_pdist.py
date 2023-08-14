import numpy as np

from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_prey import PS_1st_order_prey_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_prey import PS_1st_order_predator_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_predator import PS_1st_order_prey_on_predator
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_predator import PS_1st_order_predator_on_predator
from SOD_Utils.eval_basis_functions import eval_basis_functions


def find_phis_of_pdist(pdist_mat, theta_mat, sys_info, type):
    """
    Calculates \kappa_i'/N_i' * \phi_{K_i, K_i'}(|x_i - x_i'|) for either energy-based or alignment-based
    or xi-based influence function \phi's for different class-to-class interactions
    
    Args:
    pdist_mat: Pairwise distance, r_ii' = |x_i - x_i'|, of the whole system
    theta_mat: Pairwise angle matrix
    sys_info: System information dictionary
    type: phi and kappa type

    Returns:
    phis_of_pdist: \kappa_i'/N_i' * \phi_{K_i, K_i'}(|x_i - x_i'|) for i \in C_k1 and i' \in C_k2

    """
    N = sys_info['N']
    phis_of_pdist = np.zeros((N, N))
    
    # Pick a set of phis and kappa to work on
    if type == 'energy':
        phis = sys_info['phiE']
        kappa = sys_info['kappa']
    elif type == 'alignment':
        phis = sys_info['phiA']
        kappa = sys_info['kappa']
    elif type == 'xi':
        phis = sys_info['phiXi']
        kappa = sys_info['kappaXi']
    else:
        return phis_of_pdist
    
    # Go through the number of classes instead of agents
    sys_info['type_info'] = np.array(sys_info['type_info'])
    for k1 in range(1,sys_info['K']+1):
        row_ind = np.where(sys_info['type_info'] == k1)[0]
        N_k1 = len(row_ind)
        
        for k2 in range(1,sys_info['K']+1):
            col_ind = np.where(sys_info['type_info'] == k2)[0]
            N_k2 = len(col_ind)
            
            #pdist_Ck1_Ck2_mat = 0
            #pdist_Ck1_Ck2_vec = 0
            pdist_Ck1_Ck2_mat = pdist_mat[np.ix_(row_ind, col_ind)]
            pdist_Ck1_Ck2_vec = pdist_Ck1_Ck2_mat.flatten()
            
            if 'has_theta' in sys_info and sys_info['has_theta']:
                theta_Ck1_Ck2_mat = theta_mat[np.ix_(row_ind, col_ind)]
                theta_Ck1_Ck2_vec = theta_Ck1_Ck2_mat.flatten()
                phi_of_pdist_Ck1_Ck2 = np.reshape(phis[(k1 -1,k2 -1)](pdist_Ck1_Ck2_vec, theta_Ck1_Ck2_vec), (N_k1, N_k2))
            else:
                if len(phis) == 4:
                   if str(phis[(k1 -1,k2 -1)]).find('scipy') != -1:
                    phi_of_pdist_Ck1_Ck2 = np.reshape(phis[(k1 -1,k2 -1)](pdist_Ck1_Ck2_vec), (N_k1, N_k2))
                   else:
                       phi_of_pdist_Ck1_Ck2 = np.reshape(eval(phis[(k1 -1,k2 -1)])(pdist_Ck1_Ck2_vec), (N_k1, N_k2))
                       
                       
                else:
                    phi_of_pdist_Ck1_Ck2 = 0
            
            if N_k2 == 0:
                print("From find_phis_of_pdist\n")
                print(phis[(k1 -1,k2 -1)])
            phis_of_pdist[np.ix_(row_ind, col_ind)] = phi_of_pdist_Ck1_Ck2 * (kappa[k2-1] / N_k2)
    
    # Exception check
    np.fill_diagonal(phis_of_pdist, 0)
    ind = np.isinf(phis_of_pdist)
    #if np.sum(ind) > 0:
       # raise Exception('phi(0) = Inf!!')
       #phis_of_pdist = 0
    #ind = np.isnan(phis_of_pdist)
    #if np.sum(ind) > 0:
        #raise Exception('phi(0) = NaN!!')
        #phis_of_pdist = 0
    
    return phis_of_pdist
