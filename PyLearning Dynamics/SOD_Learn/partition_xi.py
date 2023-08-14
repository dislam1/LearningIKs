import numpy as np
from scipy.spatial.distance import pdist, squareform

def partition_xi(x, xi, sys_info):
    L = xi.shape[1]
    N = sys_info['N']
    d = sys_info['d']
    xi_regulator = sys_info['Rxi']
    type_info = sys_info['type_info']
    
    energy_pdist = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    xi_pdiff = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    xi_reg = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    num_agents_each_class = np.zeros(sys_info['K'])
    agents_class_indicator = [None] * sys_info['K']
    Psi_row_ind = [None] * sys_info['K']
    
    for l in range(L):
        xi_at_t = xi[:, l]
        x_at_t = x[:, l].reshape((d, N))
        energy_pdist_at_t = squareform(pdist(x_at_t.T))
        
        if xi_regulator is not None:
            xi_regulation = xi_regulator(xi_at_t, x_at_t)
        else:
            xi_regulation = None
        
        xi_pdiff_at_t = find_pair_diff(xi_at_t)
        
        for k_1 in range(sys_info['K']):
            if l == 0 and k_1 == 0:
                agents_Ck1 = np.where(type_info == k_1)[0]
                num_agents_Ck1 = len(agents_Ck1)
                agents_class_indicator[k_1] = agents_Ck1
                num_agents_each_class[k_1] = num_agents_Ck1
            else:
                agents_Ck1 = agents_class_indicator[k_1]
                num_agents_Ck1 = num_agents_each_class[k_1]
            
            pdist_rows = np.arange(num_agents_Ck1) + (l * num_agents_Ck1)
            pdiff_rows = pdist_rows
            pdiff_row_ind = agents_Ck1
            
            Psi_row_ind[k_1] = np.zeros(L * num_agents_Ck1)
            Psi_row_ind[k_1][pdiff_rows] = pdiff_row_ind + (l * N)
            
            for k_2 in range(sys_info['K']):
                if l == 0:
                    if k_2 == 0:
                        agents_Ck2 = agents_class_indicator[k_2]
                        num_agents_Ck2 = num_agents_each_class[k_2]
                    else:
                        agents_Ck2 = np.where(type_info == k_2)[0]
                        num_agents_Ck2 = len(agents_Ck2)
                        agents_class_indicator[k_2] = agents_Ck2
                        num_agents_each_class[k_2] = num_agents_Ck2
                else:
                    agents_Ck2 = agents_class_indicator[k_2]
                    num_agents_Ck2 = num_agents_each_class[k_2]
                
                if l == 0:
                    energy_pdist[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                    xi_pdiff[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                    xi_reg[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                
                pdist_Ck1_Ck2 = energy_pdist_at_t[agents_Ck1][:, agents_Ck2]
                energy_pdist[k_1][k_2][pdist_rows, :] = pdist_Ck1_Ck2
                
                if xi_regulation is not None:
                    xi_reg[k_1][k_2][pdist_rows, :] = xi_regulation[agents_Ck1][:, agents_Ck2]
                
                pdiff_Ck1_Ck2 = xi_pdiff_at_t[pdiff_row_ind][:, agents_Ck2]
                xi_pdiff[k_1][k_2][pdiff_rows, :] = pdiff_Ck1_Ck2
    
    return (energy_pdist, xi_pdiff, xi_reg, Psi_row_ind)
