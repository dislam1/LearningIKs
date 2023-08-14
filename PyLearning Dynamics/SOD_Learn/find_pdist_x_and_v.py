import numpy as np
from scipy.spatial.distance import pdist, squareform

def find_pdist_x_and_v(x, v, sys_info, learning_info, agents_info):
    L = x.shape[1]
    
    if sys_info.ode_order == 2:
        has_align = bool(learning_info.Abasis_info)
    elif sys_info.ode_order == 1:
        has_align = False
    
    pdist_x = [[None] * sys_info.K for _ in range(sys_info.K)]
    pdist_v = [[None] * sys_info.K for _ in range(sys_info.K)] if has_align else []
    Rs = np.zeros((sys_info.K, sys_info.K, L))
    
    for k_1 in range(sys_info.K):
        for k_2 in range(sys_info.K):
            if k_1 == k_2:
                if agents_info.num_agents[k_1] > 1:
                    pdist_x[k_1][k_2] = np.zeros((agents_info.num_agents[k_1] * (agents_info.num_agents[k_1] - 1) // 2, L))
                    
                    if has_align:
                        pdist_v[k_1][k_2] = np.zeros((agents_info.num_agents[k_1] * (agents_info.num_agents[k_1] - 1) // 2, L))
                else:
                    pdist_x[k_1][k_2] = None
                    
                    if has_align:
                        pdist_v[k_1][k_2] = None
            else:
                pdist_x[k_1][k_2] = np.zeros((agents_info.num_agents[k_1] * agents_info.num_agents[k_2], L))
                
                if has_align:
                    pdist_v[k_1][k_2] = np.zeros((agents_info.num_agents[k_1] * agents_info.num_agents[k_2], L))
    
    pair_dist_x = np.zeros((sys_info.N, sys_info.N, L))
    
    for l in range(L):
        x_at_t = x[:, l].reshape((sys_info.d, sys_info.N))
        pair_dist_x[:, :, l] = squareform(pdist(np.transpose(x_at_t)))
    
    if has_align:
        pair_dist_v = np.zeros((sys_info.N, sys_info.N, L))
        
        for l in range(L):
            v_at_t = v[:, l].reshape((sys_info.d, sys_info.N))
            pair_dist_v[:, :, l] = squareform(pdist(np.transpose(v_at_t)))
    
    for l in range(L):
        for k_1 in range(sys_info.K):
            for k_2 in range(sys_info.K):
                pair_dist_x_Ck1_Ck2 = pair_dist_x[agents_info.idxs[k_1], agents_info.idxs[k_2], l]
                
                if has_align:
                    pair_dist_v_Ck1_Ck2 = pair_dist_v[agents_info.idxs[k_1], agents_info.idxs[k_2], l]
                
                if k_1 == k_2:
                    if agents_info.num_agents[k_1] > 1:
                        pdist_x[k_1][k_2][:, l] = squareform(pair_dist_x_Ck1_Ck2)
                        
                        if has_align:
                            pdist_v[k_1][k_2][:, l] = squareform(pair_dist_v_Ck1_Ck2)
                else:
                    pdist_x[k_1][k_2][:, l] = pair_dist_x_Ck1_Ck2.flatten()
                    
                    if has_align:
                        pdist_v[k_1][k_2][:, l] = pair_dist_v_Ck1_Ck2.flatten()
                
                Rs[k_1, k_2, l] = np.max(pair_dist_x_Ck1_Ck2)
    
    Rs = np.max(Rs, axis=2)
    Rs[Rs == 0] = 1
    
    return pdist_x, pdist_v, Rs
