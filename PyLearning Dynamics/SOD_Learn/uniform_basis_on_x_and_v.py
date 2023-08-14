from SOD_Learn.uniform_basis_by_class import uniform_basis_by_class

def uniform_basis_on_x_and_v(Rs, sys_info, learn_info):
    Ebasis, Abasis = [], []
    
    # Construct energy-based interaction basis
    if learn_info['Ebasis_info']:
        Ebasis = uniform_basis_by_class(Rs, sys_info['K'], learn_info['Ebasis_info'])
    
    # Construct alignment-based interaction basis
    if sys_info['ode_order'] == 2 and learn_info['Abasis_info']:
        Abasis = uniform_basis_by_class(Rs, sys_info['K'], learn_info['Abasis_info'])
    
    return Ebasis, Abasis
