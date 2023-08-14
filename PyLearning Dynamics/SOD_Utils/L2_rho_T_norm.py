def L2_rho_T_norm(f, sys_info, rhoLT, basis, kind):
    if not isinstance(f, list):
        g = [[f for _ in range(sys_info.K)] for _ in range(sys_info.K)]
        f = g
        del g

    L2rhoTnorm = [[0 for _ in range(sys_info.K)] for _ in range(sys_info.K)]
    if kind == 'energy':
        rhoT = rhoLT['rhoLTE']
    elif kind == 'alignment':
        rhoT = rhoLT['rhoLTA']
    elif kind == 'xi':
        rhoT = rhoLT['rhoLTXi']
    else:
        raise ValueError('Invalid kind. It should be one of "energy", "alignment", or "xi".')

    for k1 in range(sys_info.K):
        for k2 in range(sys_info.K):
            if k1 == k2 and sum(sys_info['type_info'] == k1) == 1:
                L2rhoTnorm[k1][k2] = 0
            else:
                range_k1k2 = basis[k1][k2]['knots'][[0, -1]]
                range_k1k2 = intersect_interval(rhoT['supp'][k1][k2], range_k1k2)
                edges = rhoT['histedges'][k1][k2]
                e_idxs = [i for i, x in enumerate(edges) if range_k1k2[0] <= x < range_k1k2[1]]
                centers = [(edges[i] + edges[i + 1]) / 2 for i in e_idxs[:-1]]
                weights = centers
                histdata = rhoT['hist'][k1][k2][e_idxs[:-1]] if 'hist' in rhoT else None
                f_vec = [f[k1][k2](x) for x in centers]
                f_integrand = [(x * y) ** 2 * z for x, y, z in zip(f_vec, weights, histdata)] if histdata else [(x * y) ** 2 for x, y in zip(f_vec, weights)]
                L2rhoTnorm[k1][k2] = sum(f_integrand) ** 0.5

    return L2rhoTnorm
