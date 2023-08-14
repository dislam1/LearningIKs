import numpy as np

def solve_for_interactions_on_xi_by_others(Phi, rhs_vec, basis, T, M, learn_info):
    solver_type = learn_info['solver_type']
    if solver_type == 'linsolve':
        opts = {'SYM': True}
        alpha_vec = np.linalg.solve(Phi, rhs_vec, **opts)
    elif solver_type == 'mldivide':
        alpha_vec = np.linalg.solve(Phi, rhs_vec)
    elif solver_type == 'inverse':
        alpha_vec = np.linalg.inv(Phi) @ rhs_vec
    elif solver_type == 'pinv':
        alpha_vec = np.linalg.pinv(Phi) @ rhs_vec
    else:
        raise ValueError('Other types of solvers are not implemented yet!')

    if T == 0:
        opt_val = (Phi @ alpha_vec).T @ alpha_vec - 2 * alpha_vec.T @ rhs_vec / M
    else:
        opt_val = (Phi @ alpha_vec).T @ alpha_vec - 2 * alpha_vec.T @ rhs_vec / (T * M)

    return alpha_vec, opt_val
