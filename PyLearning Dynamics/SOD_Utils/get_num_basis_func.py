import numpy as np

def get_num_basis_func(sys_info):
    if sys_info['name'] == 'Gravitation':
        Rs = [0, 60, 110, 150, 230, 780, 2880, 4500, 5910]
        Rs_cut = Rs[:sys_info['N']]
        max_Rs = np.tile(Rs_cut, (sys_info['N'], 1)) + np.tile(Rs_cut, (sys_info['N'], 1)).T
        ns = 2 * max_Rs
        ind = np.eye(sys_info['N'], dtype=bool)
        ns[ind] = 1
    else:
        raise ValueError('Other systems are not yet implemented!!')
    return ns
