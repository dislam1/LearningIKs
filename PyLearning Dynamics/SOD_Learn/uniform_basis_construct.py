def uniform_basis_construct(Rs, sys_info, learn_info):
    Ebasis, Abasis, Xibasis = [], [], []
    
    # Construct energy-based interaction basis
    if learn_info['Ebasis_info']:
        Ebasis = uniform_basis_by_class(Rs, sys_info['K'], learn_info['Ebasis_info'])
    
    # for second order system
    if sys_info['ode_order'] == 2:
        # Construct alignment-based interaction basis
        if learn_info['Abasis_info']:
            Abasis = uniform_basis_by_class(Rs, sys_info['K'], learn_info['Abasis_info'])
        
        # Construct xi-based interaction basis
        if sys_info['has_xi']:
            Xibasis = uniform_basis_by_class(Rs, sys_info['K'], learn_info['Xibasis_info'])
    
    return Ebasis, Abasis, Xibasis
