import numpy as np
from SOD_Utils.sqdist_mod import sqdist_mod

def partition_traj(traj, sys_info):
    def squareform_tovector(Y):
        n = Y.shape[1]
        Z = Y[np.tril(np.ones((n, n), dtype=bool), -1)]
        Z = Z.ravel()  # force to a row vector, even if empty
        return Z

    if sys_info['ode_order'] == 2:
        has_align = len(sys_info['phiA']) > 0
        has_xi = sys_info['has_xi']
    else:
        has_align = False
        has_xi = False

    L = traj.shape[1]
    all_max_rs = np.zeros((sys_info['K'], sys_info['K'], L))
    all_min_rs = np.zeros((sys_info['K'], sys_info['K'], L))
    num_agents_each_type = np.zeros(sys_info['K'])
    type_ind = [None] * sys_info['K']
    pdist_x = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]

    if has_align:
        pdist_v = [[None for _ in range(sys_info.K)] for _ in range(sys_info.K)]
        all_max_rdots = np.zeros((sys_info.K, sys_info.K, L))
        all_min_rdots = np.zeros((sys_info.K, sys_info.K, L))
    else:
        pdist_v = []

    if has_xi:
        pdist_xi = [[None for _ in range(sys_info.K)] for _ in range(sys_info.K)]
        all_max_xis = np.zeros((sys_info.K, sys_info.K, L))
        all_min_xis = np.zeros((sys_info.K, sys_info.K, L))
    else:
        pdist_xi = []

       
    block_size = sys_info['d'] * sys_info['N']

    for l in range(L):
        x_at_t = traj[0:block_size, l].reshape(sys_info['d'], sys_info['N'])
        #print(x_at_t.shape)
        pdist_x_at_t = np.sqrt(sqdist_mod(x_at_t, x_at_t))

        if has_align:
            v_at_t = traj[block_size:2 * block_size, l].reshape(sys_info['d'], sys_info['N'])
            pdist_v_at_t = np.sqrt(sqdist_mod(v_at_t, v_at_t))

        if has_xi:
            xi_at_t = traj[2 * block_size:2 * block_size + sys_info['N'], l].reshape(1, sys_info['N'])
            pdist_xi_at_t = np.sqrt(sqdist_mod(xi_at_t, xi_at_t))

        for k_1 in range(sys_info['K']):
            if l == 0 and k_1 == 0:
                agents_Ck1 = np.where(sys_info['type_info'] == k_1+1)[0]
                type_ind[k_1] = agents_Ck1
                num_agents_Ck1 = len(agents_Ck1)
                num_agents_each_type[k_1] = num_agents_Ck1
            else:
                agents_Ck1 = type_ind[k_1]
                num_agents_Ck1 = int(num_agents_each_type[k_1])

            for k_2 in range(sys_info['K']):
                if l == 0:
                    if k_2 == 0:
                        agents_Ck2 = np.where(sys_info['type_info'] == k_2+1)[0]
                        type_ind[k_2] = agents_Ck2
                        num_agents_Ck2 = len(agents_Ck2)
                        num_agents_each_type[k_2] = num_agents_Ck2
                    else:
                        agents_Ck2 = np.where(sys_info['type_info'] == k_2+1)[0]
                        type_ind[k_2] = agents_Ck2
                        num_agents_Ck2 = len(agents_Ck2)
                        num_agents_each_type[k_2] = num_agents_Ck2
                else:
                    agents_Ck2 = type_ind[k_2]
                    num_agents_Ck2 = int(num_agents_each_type[k_2])

                if l == 0:
                    if k_1 == k_2:
                        if num_agents_Ck1 > 1:
                            pdist_x[k_1][k_2] = np.zeros((num_agents_Ck1 * (num_agents_Ck1 - 1) // 2, L))
                            if has_align:
                                pdist_v[k_1][k_2] = np.zeros((num_agents_Ck1 * (num_agents_Ck1 - 1) // 2, L))
                            if has_xi:
                                pdist_xi[k_1][k_2] = np.zeros((num_agents_Ck1 * (num_agents_Ck1 - 1) // 2, L))
                        else:
                            pdist_x[k_1][k_2] = None
                            if has_align:
                                pdist_v[k_1][k_2] = None
                            if has_xi:
                                pdist_xi[k_1][k_2] = None
                    else:
                        #print('Trajectory')
                        #print(pdist_x)
                        #print(num_agents_Ck1)
                        #print(num_agents_Ck2)
                        pdist_x[k_1][k_2] = np.zeros((num_agents_Ck1 * num_agents_Ck2, L))
                        if has_align:
                            pdist_v[k_1][k_2] = np.zeros((num_agents_Ck1 * num_agents_Ck2, L))
                        if has_xi:
                            pdist_xi[k_1][k_2] = np.zeros((num_agents_Ck1 * num_agents_Ck2, L))

                pdist_x_Ck1_Ck2 = pdist_x_at_t[agents_Ck1][:, agents_Ck2]
                if has_align:
                    pdist_v_Ck1_Ck2 = pdist_v_at_t[agents_Ck1][:, agents_Ck2]
                if has_xi:
                    pdist_xi_Ck1_Ck2 = pdist_xi_at_t[agents_Ck1][:, agents_Ck2]

                if k_1 == k_2:
                    if num_agents_Ck1 > 1:
                        pdist_x[k_1][k_2][:, l] = squareform_tovector(pdist_x_Ck1_Ck2)
                        if has_align:
                            pdist_v[k_1][k_2][:, l] = squareform_tovector(pdist_v_Ck1_Ck2)
                        if has_xi:
                            pdist_xi[k_1][k_2][:, l] = squareform_tovector(pdist_xi_Ck1_Ck2)
                else:
                    pdist_x[k_1][k_2][:, l] = pdist_x_Ck1_Ck2.ravel()
                    if has_align:
                        pdist_v[k_1][k_2][:, l] = pdist_v_Ck1_Ck2.ravel()
                    if has_xi:
                        pdist_xi[k_1][k_2][:, l] = pdist_xi_Ck1_Ck2.ravel()

                if pdist_x[k_1][k_2] is not None:
                    all_max_rs[k_1, k_2, l] = np.max(pdist_x[k_1][k_2][:, l])
                    all_min_rs[k_1, k_2, l] = np.min(pdist_x[k_1][k_2][:, l])
                if has_align and pdist_v[k_1][k_2] is not None:
                    all_max_xis[k_1, k_2, l] = np.max(pdist_v[k_1][k_2][:, l])
                    all_min_xis[k_1, k_2, l] = np.min(pdist_v[k_1][k_2][:, l])
                if has_xi and pdist_xi[k_1][k_2] is not None:
                    all_max_xis[k_1, k_2, l] = np.max(pdist_xi[k_1][k_2][:, l])
                    all_min_xis[k_1, k_2, l] = np.min(pdist_xi[k_1][k_2][:, l])

    max_r = np.max(all_max_rs, axis=2)
    max_r[max_r == 0] = 1
    min_r = np.min(all_min_rs, axis=2)
    if has_align:
        max_rdot = np.max(all_max_rdots, axis=2)
        max_rdot[max_rdot == 0] = 1
        min_rdot = np.min(all_min_rdots, axis=2)
    else:
        max_rdot = []
        min_rdot = []
    if has_xi:
        max_xi = np.max(all_max_xis, axis =2)
        max_xi[max_xi == 0] = 1
        min_xi = np.min(all_min_xis, axis = 2)
    else:
        max_xi = []
        min_xi = []
    #print("Max r shape")
    #print(max_r.shape)
    pdist_info = {
        'pdist_x': pdist_x,
        'pdist_v': pdist_v,
        'pdist_xi': pdist_xi,
        'max_r': max_r,
        'min_r': min_r,
        'max_rdot': max_rdot,
        'min_rdot': min_rdot
    }

    return pdist_info


# Assuming traj and sys_info are available as inputs to this function.

# Example usage:
# pdist_info = partition_traj(traj, sys_info)
