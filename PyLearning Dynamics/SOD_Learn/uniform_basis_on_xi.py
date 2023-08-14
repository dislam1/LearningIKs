def uniform_basis_on_xi(Rs, sys_info, learn_info):
    basis = uniform_basis_by_class(Rs, sys_info['K'], learn_info['Xibasis_info'])
    return basis
