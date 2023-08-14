import numpy as np
from scipy.spatial.distance import pdist, squareform
from SOD_Evolve.find_pair_diff import find_pair_diff

def partition_x_and_v(x, v, sys_info, learn_info):
    L = x.shape[1]
    N = sys_info['N']
    d = sys_info['d']
    K = sys_info['K']
    type_info = sys_info['type_info']
    ode_order = sys_info['ode_order']
    has_energy = bool(learn_info['Ebasis_info'])
    
    if ode_order == 1:
        has_align = False
    else:
        has_align = bool(learn_info['Abasis_info'])
    
    if ode_order == 2 and has_align:
        align_regulator                                   = sys_info['RA']
        
    energy_regulator = sys_info['RE']
    energy_pdist = np.array([[None]  for _ in range(K)]*K).reshape(K,K)
    
    energy_pdiff = np.array([[None]  for _ in range(K)]*K).reshape(K,K) if has_energy else []
    energy_reg = np.array([[None]  for _ in range(K)]*K).reshape(K,K) if has_energy else []
    align_pdist = np.array([[None]  for _ in range(K)]*K).reshape(K,K) if has_align else []
    align_pdiff = np.array([[None]  for _ in range(K)]*K).reshape(K,K) if has_align else []
    align_reg = np.array([[None]  for _ in range(K)]*K).reshape(K,K) if has_align else []
    Psi_row_ind = [None] *K
    
    num_agents_each_class = np.zeros(K, dtype=np.int32)
    agents_class_indicator = [None] * K
    row_ind_Phi_all_class = [None] * K
    
    for l in range(L):
        x_at_t = x[:, l].reshape((d, N))
        
        if ode_order == 1:
            v_at_t = None
        else:
            v_at_t = v[:, l].reshape((d, N))
        
        x_pdist = squareform(pdist(x_at_t.T))
        
        if has_energy:
            #x_pdiff = np.zeros((d * N, N))
            x_pdiff = find_pair_diff(x_at_t)
            
            if energy_regulator is not None:
                energy_regulation = energy_regulator(x_at_t)
            else:
                energy_regulation = None
            #for i in range(N):
                #x_pdiff[i * d : (i + 1) * d, :] = x_at_t[:, i].reshape((-1, 1)) - x_at_t
            #x_pdiff = x_pdiff.flatten()
        
        if has_align:
            #v_pdiff = np.zeros((d * N, N))
            v_pdiff = find_pair_diff(v_at_t)
            align_regulation = sys_info['RA'](x_at_t, v_at_t) if sys_info['RA'] else None
            
            if align_regulator is not None:
                align_regulation = align_regulator(x_at_t, v_at_t)
            else:
                align_regulation = None
            #for i in range(N):
                #v_pdiff[i * d : (i + 1) * d, :] = v_at_t[:, i].reshape((-1, 1)) - v_at_t
            #v_pdiff = v_pdiff.flatten()
            v_pdist = squareform(pdist(v_at_t.T))
        
        for k_1 in range(K):
            if l == 0 and k_1 == 0:
                agents_Ck1 = np.where(type_info == (k_1+1))[0]
                agents_class_indicator[k_1] = agents_Ck1
                num_agents_Ck1 = len(agents_Ck1)
                num_agents_each_class[k_1] = num_agents_Ck1
            else:
                agents_Ck1 = agents_class_indicator[k_1]
                num_agents_Ck1 = num_agents_each_class[k_1]
            
            row_ind_in_pdist = (np.arange(num_agents_Ck1)).reshape(num_agents_Ck1,1) + (l * num_agents_Ck1)
            row_ind_in_pdiff = (np.arange(num_agents_Ck1 * d)).reshape(num_agents_Ck1 * d,1) + (l * num_agents_Ck1 * d)
            
            if l == 0:
                row_ind_Phi = np.asarray(np.tile(np.arange(d).reshape((-1, 1)), (num_agents_Ck1,1)),dtype=np.int64)
                row_ind_Phi += np.asarray(np.kron(((agents_Ck1.reshape((-1, 1))) ) * d, np.ones((d, 1))),dtype=np.int64)
                row_ind_Phi_all_class[k_1] = row_ind_Phi
            else:
                row_ind_Phi = row_ind_Phi_all_class[k_1]
            #the row indices in the Phi matrices    
            if l == 0:
                Psi_row_ind[k_1] = np.zeros(L * num_agents_Ck1 * d, dtype=np.int32)
                        
            Psi_row_ind[k_1][row_ind_in_pdiff] = (row_ind_Phi + (l * N * d))
            
            for k_2 in range(K):
                if l == 0:
                    if k_2 == 0:
                        agents_Ck2 = agents_class_indicator[k_2]
                        num_agents_Ck2 = num_agents_each_class[k_2]
                    else:
                        agents_Ck2 = np.where(type_info == k_2+1)[0]
                        agents_class_indicator[k_2] = agents_Ck2
                        num_agents_Ck2 = len(agents_Ck2)
                        num_agents_each_class[k_2] = num_agents_Ck2
                else:
                    agents_Ck2 = agents_class_indicator[k_2]
                    num_agents_Ck2 = num_agents_each_class[k_2]
                
                if l == 0:
                    #energy_pdist[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2,len(row_ind_in_pdist),len(row_ind_in_pdist) ))
                    energy_pdist[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                    #energy_pdist[k_1][k_2] =[[0 for m in range(num_agents_Ck2)] for w in range(L * num_agents_Ck1)]
                    if has_energy:
                        energy_pdiff[k_1][k_2] = np.zeros((L * num_agents_Ck1 * d, num_agents_Ck2))
                        energy_reg[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                    if has_align:
                        #align_pdist[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2, len(row_ind_in_pdist)  , len(agents_Ck2)))
                        #align_pdiff[k_1][k_2] = np.zeros((L * num_agents_Ck1 * d, num_agents_Ck2,len(row_ind_in_pdiff),len(agents_Ck2)))
                        #align_reg[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2, len(row_ind_in_pdiff), len(agents_Ck2)))
                        
                        align_pdist[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                        align_pdiff[k_1][k_2] = np.zeros((L * num_agents_Ck1 * d, num_agents_Ck2))
                        align_reg[k_1][k_2] = np.zeros((L * num_agents_Ck1, num_agents_Ck2))
                
                x_pdist_Ck1_Ck2 = x_pdist[agents_Ck1][:, agents_Ck2]
                for i, j in enumerate(row_ind_in_pdist):             
                    energy_pdist[k_1][k_2][j, :] = x_pdist_Ck1_Ck2[i]
                
                
                if has_energy:
                    if energy_regulation is not None:
                        energy_mat = energy_regulation[agents_Ck1][:, agents_Ck2]
                        for i, j in enumerate(row_ind_in_pdist): 
                            energy_reg[k_1][k_2][j,:] = energy_mat[i]
                    else:
                        if l==0:
                            energy_reg[k_1][k_2]=[]
                    x_pdiff_Ck1_Ck2                             = x_pdiff[row_ind_Phi, agents_Ck2] 
                    for i, j in enumerate(row_ind_in_pdiff):       
                        energy_pdiff[k_1][k_2][j,:] = x_pdiff_Ck1_Ck2[i]
                
                if has_align:
                    vpid_mat = v_pdist[agents_Ck1,agents_Ck2]
                    for i, j in enumerate(row_ind_in_pdist): 
                        align_pdist[k_1][k_2][j,:] = vpid_mat[i]
                        
                    if align_regulation is not None:
                        align_mat = align_regulation[agents_Ck1,agents_Ck2]
                        for i, j in enumerate(row_ind_in_pdiff):
                            align_reg[k_1][k_2][j,:] = align_mat[i]
                    else:
                        if l==0:
                            align_reg[k_1][k_2] = []
                    v_pdiff_mat = v_pdiff[row_ind_Phi, agents_Ck2]
                    align_pdiff[k_1][k_2][row_ind_in_pdiff,:] = v_pdiff_mat[row_ind_in_pdiff]
    
    return (energy_pdist, energy_pdiff, energy_reg, align_pdist, align_pdiff, align_reg, Psi_row_ind)
