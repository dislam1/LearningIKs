from SOD_Utils.eval_basis_functions import eval_basis_functions
import numpy as np
def LinearCombinationBasis(basis, alpha):
    basis = np.array(basis)
    if basis.size == 0:
        f_ptr = []
        lastIdx = 0
        return f_ptr, lastIdx
    
    num_classes = basis.shape[0]
    f_ptr = np.empty((num_classes, num_classes), dtype=object)
    
    sum_prev_num_basis = 0
    for k_1 in range(1, num_classes+1):  # go through each class-to-class interaction
        for k_2 in range(1, num_classes+1):
            lamfun = {}
            one_basis = basis[k_1-1][k_2-1]  # basis for C_{s_1} to C_{s_2} interaction
            num_basis = len(one_basis['f'])  # number of basis functions
            ind_1 = sum_prev_num_basis  # the starting index to cut alpha_vec
            ind_2 = sum_prev_num_basis + num_basis  # the ending index to cut alpha_vec
            one_alphas = alpha[ind_1:ind_2]  # portion of the alpha_vec corresponding to this interaction
            #f_ptr[k_1-1, k_2-1] = 'lambda r, alpha_vec='+ str(one_alphas)+', basis='+ str(one_basis)+ ': eval_basis_functions(r, alpha_vec, basis)'  # the sum_{l=1}^L \alpha_l \phi_l is the learned interaction
            f_ptr[k_1-1][k_2-1] = {'alpha_vec':one_alphas, 'basis':one_basis, 'fun':'eval_basis_functions(r, alpha_vec, basis)'}
            sum_prev_num_basis += num_basis  # update the sum of previous number of basis functions
    
    lastIdx = sum_prev_num_basis
    
    return f_ptr, lastIdx
