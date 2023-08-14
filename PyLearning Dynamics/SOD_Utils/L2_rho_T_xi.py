def L2_rho_T_xi(f, sys_info, obs_info, basis):
    if not isinstance(f, list):
        g = [[f for _ in range(sys_info.K)] for _ in range(sys_info.K)]
        f = g
        del g

    L2rhoTnorm = [[0 for _ in range(sys_info.K)] for _ in range(sys_info.K)]
    rhoLTXi = obs_info['rhoLT']['rhoLTXi']

    for k1 in range(sys_info.K):
        N_k1 = sum(sys_info['type_info'] == k1)
        for k2 in range(sys_info.K):
            if k1 == k2 and N_k1 == 1:
                L2rhoTnorm[k1][k2] = 0
            else:
                range_k1k2 = basis[k1][k2]['knots'][[0, -1]]
                range_k1k2 = intersect_interval(rhoLTXi['supp'][k1][k2][0], range_k1k2)
                edges_R = rhoLTXi['histedges'][k1][k2][0]
                e_idxs = [i for i, x in enumerate(edges_R) if range_k1k2[0] <= x < range_k1k2[1]]
                centers = [(edges_R[i] + edges_R[i + 1]) / 2 for i in e_idxs[:-1]]
                range_Xi = rhoLTXi['supp'][k1][k2][1]
                edges_Xi = rhoLTXi['histedges'][k1][k2][1]
                wgt_idxs = [i for i, x in enumerate(edges_Xi) if range_Xi[0] <= x < range_Xi[1]]
                weights = [(edges_Xi[i] + edges_Xi[i + 1]) / 2 for i in wgt_idxs[:-1]]
                histdata = rhoLTXi['hist'][k1][k2][e_idxs[:-1], wgt_idxs[:-1]] if 'hist' in rhoLTXi else None
                f_vec = [f[k1][k2](x) for x in centers]
                f_integrand = [((x * y) ** 2) * z for x, y, z in zip(f_vec, weights, histdata)] if histdata else [(x * y) ** 2 for x, y in zip(f_vec, weights)]
                L2rhoTnorm[k1][k2] = sum(f_integrand) ** 0.5

    return L2rhoTnorm
