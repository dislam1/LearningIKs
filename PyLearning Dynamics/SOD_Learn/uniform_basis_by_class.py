import numpy as np
from SOD_Learn.uniform_basis import uniform_basis

def uniform_basis_by_class(Rs, K, basis_info):
    if Rs.shape[0] < K:
        Rs = Rs * np.ones((K, K))
    print('From uniform_basis_by_class \n')
    print(Rs)
    
    if basis_info:
        basis = [[None for _ in range(K)] for _ in range(K)]
        for k_1 in range(K):
            for k_2 in range(K):
                b = Rs[k_1, k_2]
                p = basis_info['degree'][k_1, k_2]
                num_basis_funs = int(basis_info['n'][k_1, k_2])
                basis[k_1][k_2] = uniform_basis(b, p, num_basis_funs, basis_info)
                basis[k_1][k_2]['degree'] = p
                basis[k_1][k_2]['Rmax'] = Rs[k_1, k_2]  # TBD
    else:
        basis = []
    #print(basis)
    return basis
