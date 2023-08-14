import numpy as np

def assemble_the_learning_matrix_on_xi(energy_pdist, xi_pdiff, xi_regulator, xi_basis, time_vec, agents_info, Psi_row_ind, sys_info, learn_info):
    # function Psi = assemble_the_learning_matrix_on_xi(energy_pdist, xi_pdiff, xi_regulator, xi_basis, time_vec, agents_info, Psi_row_ind, learn_info)
    # (c) M. Zhong (JHU)

    # find out the number of basis functions for each class-to-class interaction
    num_basis_each_class = np.array([len(x['f']) for x in xi_basis])
    total_num_basis = np.sum(num_basis_each_class)
    
    # allocate the storage for the learning matrix Psi
    L = len(time_vec)
    Psi = np.zeros((sys_info['N'] * L, total_num_basis))
    
    # remember the number of basis functions from previous class-to-class interaction
    num_prev_basis = 0
    
    # go through each Cs1-Cs2 interaction
    for k_1 in range(1, sys_info['K']+1):
        # the row indices in the Psi matrix
        row_ind = Psi_row_ind[k_1-1]
        
        for k_2 in range(1, sys_info['K']+1):
            # find out the number of basis functions in this C_s1-C_s2 interaction
            num_basis = num_basis_each_class[k_1-1, k_2-1]
            
            # the starting column index to put class_influence in the Psi matrix
            ind_1 = num_prev_basis
            ind_2 = num_prev_basis + num_basis
            
            # update the number of basis functions from previous Cs1-Cs2 interaction
            num_prev_basis += num_basis
            
            if k_1 == k_2 and agents_info['num_agents'][k_2-1] == 1:
                # for (C_k1, C_k1) interaction, if the class has only one agent, no influence at all
                Psi[row_ind, ind_1:ind_2] = 0
            else:
                # find out the influence from basis functions: phi_l(|x_j - x_i|)(x_j - x_i)
                # or phi_l(|x_j - x_i|)(xi_j - xi_i) for i in C_s1, and j in C_s2
                Psi[row_ind, ind_1:ind_2] = find_class_influence(xi_basis[k_1-1][k_2-1], energy_pdist[k_1-1][k_2-1],
                                                                  xi_regulator[k_1-1][k_2-1], xi_pdiff[k_1-1][k_2-1],
                                                                  1, agents_info['num_agents'][k_2-1],
                                                                  sys_info['kappaXi'][k_2-1], learn_info['Xibasis_info'])
    return Psi

'''
Note: The code assumes that you have defined the find_class_influence() function. 
You'll need to provide the definition for this function to run the code successfully.
'''