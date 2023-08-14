from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_prey import PS_1st_order_prey_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_prey import PS_1st_order_predator_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_predator import PS_1st_order_prey_on_predator
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_predator import PS_1st_order_predator_on_predator
from SOD_Utils.eval_basis_functions import eval_basis_functions

from SOD_Utils.intersectInterval import intersectInterval
import numpy as np

def L2_rho_T_energy(f, sys_info, obs_info, basis):
    if not isinstance(f,  dict):
        g = [[f for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        f = g
        del g

    L2rhoTnorm = [[0 for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    rhoLTE = obs_info['rhoLT']['rhoLTE']

    for k1 in range(sys_info['K']):
        N_k1 = np.count_nonzero(sys_info['type_info'] == k1+1)
        for k2 in range(sys_info['K']):
            if k1 == k2 and N_k1 == 1:
                L2rhoTnorm[k1][k2] = 0
            else:
                range_k1k2 = basis[k1][k2]['knots'][[0, -1]]
                range_k1k2 = intersectInterval(rhoLTE['supp'][k1][k2], range_k1k2)
                edges = rhoLTE['histedges'][k1][k2]
                e_idxs = [i for i, x in enumerate(edges) if range_k1k2[0] <= x < range_k1k2[1]]
                centers = [(edges[i] + edges[i + 1]) / 2 for i in e_idxs[:-1]]
                weights = centers
                histdata = rhoLTE['hist'][k1][k2][e_idxs[:-1]]
                if 'lc' in f[(k1,k2)]:
                    if isinstance(f[(k1,k2)]['lc'], dict) and 'alpha_vec' in f[(k1,k2)]['lc']:
                    #Calculate the difference
                        f_ps = eval(f[(k1,k2)]['PS'])(centers)
                        alpha_vec = f[(k1,k2)]['lc']['alpha_vec']
                        f_basis = f[(k1,k2)]['lc']['basis']
                        fun = f[(k1,k2)]['lc']['fun']
                        f_eval = (lambda r : eval_basis_functions(r, alpha_vec, f_basis) )(centers)
                        f_vec = f_ps - f_eval
                    else:
                        f_ps = eval(f[(k1,k2)]['PS'])(centers)
                        f_eval =f[(k1,k2)]['lc'](centers)
                        f_vec = f_ps - f_eval
                else:
                    f_vec = eval(f[(k1,k2)])(centers)
                     
                f_integrand = [(x * y) ** 2 * z for x, y, z in zip(f_vec, weights, histdata)]
                L2rhoTnorm[k1][k2] = sum(f_integrand) ** 0.5

    return L2rhoTnorm
