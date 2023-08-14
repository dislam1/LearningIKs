def L2_rho_T_alignment(f, sys_info, obs_info, basis):
    if not isinstance(f, list):
        g = [[f for _ in range(sys_info.K)] for _ in range(sys_info.K)]
        f = g
        del g

    L2rhoTnorm = [[0 for _ in range(sys_info.K)] for _ in range(sys_info.K)]
    rhoLTA = obs_info['rhoLT']['rhoLTA']

    for k1 in range(sys_info.K):
        N_k1 = sum(sys_info['type_info'] == k1)
        for k2 in range(sys_info.K):
            if k1 == k2 and N_k1 == 1:
                L2rhoTnorm[k1][k2] = 0
            else:
                range_k1k2 = basis[k1][k2]['knots'][[0, -1]]
                range_k1k2 = intersect_interval(rhoLTA['supp'][k1][k2][0], range_k1k2)
                edges_R = rhoLTA['histedges'][k1][k2][0]
                ctr_idxs = [i for i, x in enumerate(edges_R) if range_k1k2[0] <= x < range_k1k2[1]]
                centers = [(edges_R[i] + edges_R[i + 1]) / 2 for i in ctr_idxs[:-1]]
                range_k1k2 = rhoLTA['supp'][k1][k2][1]
                edges_DR = rhoLTA['histedges'][k1][k2][1]
                wgt_idxs = [i for i, x in enumerate(edges_DR) if range_k1k2[0] <= x < range_k1k2[1]]
                weights = [(edges_DR[i] + edges_DR[i + 1]) / 2 for i in wgt_idxs[:-1]]
                histdata = [rhoLTA['hist'][k1][k2][i][:, wgt_idxs[:-1]] for i in ctr_idxs[:-1]]
                f_vec = [f[k1][k2](x) for x in centers]
                f_integrand = [(sum(x * weights) ** 2) * y for x, y in zip(f_vec, histdata)]
                L2rhoTnorm[k1][k2] = sum(sum(f_integrand)) ** 0.5

    return L2rhoTnorm
