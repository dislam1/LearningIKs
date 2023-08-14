
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse import spdiags
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from SOD_Learn.find_class_influence import find_class_influence

def create_sparse_zero_matrix(L,N,d,total_energy_basis):
    I = np.zeros(L * N * d) # Row
    J = np.zeros(L * N * d) #Column
    V =np.zeros(L * N * d) ##rd element
    A = sp.coo_matrix((V,(I,J)),shape=(L * N * d,total_energy_basis))
    
    return A.toarray()



def assemble_the_learning_matrix_on_x_and_v(energy_pdist, energy_pdiff, energy_regulator, energy_basis,
                                            align_pdiff, align_regulator, align_basis, time_vec,
                                            agents_info, Phi_row_ind, sys_info, learn_info):
    # function [energy_Phi, align_Phi] = assemble_the_learning_matrix_on_x_and_v(energy_pdist, ...
    # energy_pdiff, energy_regulator, energy_basis, align_pdiff, align_regulator, align_basis, ...
    # time_vec, agents_info, Phi_row_ind, sys_info)

    # (c) M. Zhong

    ISSPARSE = False

    has_energy = bool(energy_basis)
    has_align = bool(align_basis)
    L = len(time_vec)
    if has_energy:
        energy_each_class = np.array([[len(p['f']) for p in x] for x in energy_basis])
        total_energy_basis = np.sum(energy_each_class)
    if has_align:
        align_each_class = np.array([[len(p['f']) for p in x] for x in align_basis])
        total_align_basis = np.sum(align_each_class)

    # allocate the storage for the learning matrix Phi
    if has_energy:
        #energy_Phi = create_sparse_zero_matrix(L,sys_info['N'] , sys_info['d'], total_energy_basis)
        energy_Phi = csr_array((L * sys_info['N']  * sys_info['d'],total_energy_basis)).toarray()
        #energy_Phi = spdiags([[]] * (L * sys_info['N'] * sys_info['d']), total_energy_basis, 10 * L * sys_info['N'] * sys_info['d'])
    else:
        energy_Phi = []
    if has_align:
        #align_Phi = create_sparse_zero_matrix(L, sys_info['N'], sys_info['d'], total_align_basis)
        align_Phi = csr_array((L * sys_info['N']  * sys_info['d'],total_energy_basis)).toarray()
    else:
        align_Phi = []

    if not ISSPARSE:
        energy_Phi = energy_Phi
        align_Phi = align_Phi

    if has_energy:  # Energy-type interactions
        num_prev_energy = 0  # remember the number of basis functions from previous (C_k1, C_k2) interaction
        for k_1 in range(1, sys_info['K'] + 1):  # go through each (Ck1, Ck2) interaction
            for k_2 in range(1, sys_info['K'] + 1):
                ind_1 = num_prev_energy  # the starting column index to put class_influence in the Phi matrix
                ind_2 = num_prev_energy + energy_each_class[k_1 - 1, k_2 - 1]
                num_prev_energy += energy_each_class[k_1 - 1, k_2 - 1]  # update the number of basis functions from previous (C_k1, C_k2) interaction
                if k_1 == k_2 and agents_info['num_agents'][k_2 - 1] == 1:
                    energy_Phi[Phi_row_ind[k_1 - 1], ind_1:ind_2] = 0  # no interaction on a class of single agent
                else:
                    class_influence = find_class_influence(energy_basis[k_1 - 1][k_2 - 1],
                                                           energy_pdist[k_1 - 1][k_2 - 1],
                                                           energy_regulator[k_1 - 1][k_2 - 1],
                                                           energy_pdiff[k_1 - 1][k_2 - 1],
                                                           sys_info['d'], agents_info['num_agents'][k_2 - 1],
                                                           sys_info['kappa'][k_2 - 1], learn_info['Ebasis_info'], ISSPARSE)
                    #energy_Phi[Phi_row_ind[k_1 - 1], ind_1:ind_1 + class_influence.shape[1]] = class_influence
                    #for i,j in enumerate(Phi_row_ind[k_1 - 1]): energy_Phi[int(j)][ind_1][ind_1:ind_1 + class_influence.shape[1]]=class_influence[i,:]
                    energy_Phi[Phi_row_ind[k_1 - 1],ind_1:ind_1 + class_influence.shape[1]] = class_influence
    if has_align:  # do the same thing over again for alignment terms
        num_prev_align = 0  # remember the number of basis functions from previous (C_k1, C_k2) interaction
        for k_1 in range(1, sys_info['K'] + 1):  # go through each (Ck1, Ck2) interaction
            for k_2 in range(1, sys_info['K'] + 1):
                ind_1 = num_prev_align  # the starting column index to put class_influence in the Phi matrix
                ind_2 = num_prev_align + align_each_class[k_1 - 1, k_2 - 1]
                num_prev_align += align_each_class[k_1 - 1, k_2 - 1]  # update the number of basis functions from previous (C_k1, C_k2) interaction
                if k_1 == k_2 and agents_info['num_agents'][k_2 - 1] == 1:
                    align_Phi[Phi_row_ind[k_1 - 1], ind_1:ind_2] = 0  # no interaction on a class of single agent
                else:
                    class_influence = find_class_influence(align_basis[k_1 - 1][k_2 - 1],
                                                           energy_pdist[k_1 - 1][k_2 - 1],
                                                           align_regulator[k_1 - 1][k_2 - 1],
                                                           align_pdiff[k_1 - 1][k_2 - 1],
                                                           sys_info['d'], agents_info['num_agents'][k_2 - 1],
                                                           sys_info['kappa'][k_2 - 1], learn_info['Abasis_info'], ISSPARSE)
                    #align_Phi[Phi_row_ind[k_1 - 1], ind_1:ind_1 + class_influence.shape[1]] = class_influence
                    #for i,j in enumerate(Phi_row_ind[k_1 - 1]): align_Phi[int(j)][ind_1][ind_1:ind_1 + class_influence.shape[1]]=class_influence[i,:]
                    align_Phi[Phi_row_ind[k_1 - 1],ind_1:ind_1 + class_influence.shape[1]] = class_influence

    return energy_Phi, align_Phi
