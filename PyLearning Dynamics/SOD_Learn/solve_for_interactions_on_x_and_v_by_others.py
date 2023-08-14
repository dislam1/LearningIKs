import numpy as np
from scipy.linalg import solve, pinv

def solve_for_interactions_on_x_and_v_by_others(Phi, rhs_vec, T, M, learn_info):
    if 'solver_info' in learn_info:
        solver_info = learn_info['solver_info']
    else:
        solver_info = {}
    solver_type = learn_info['solver_type']

    if solver_type == 'linsolve':
        opts = {'SYM': True}
        alpha_vec = solve(Phi, rhs_vec, **opts)
    elif solver_type == 'mldivide':
        alpha_vec = solve(Phi, rhs_vec)
    elif solver_type == 'inverse':
        alpha_vec = np.dot(pinv(Phi), rhs_vec)
    elif solver_type == 'pinv':
        alpha_vec = np.dot(pinv(Phi), rhs_vec)
    elif solver_type == 'svd':
        U, S, V = np.linalg.svd(Phi)
        lambdas = np.diag(S)
        if 'svd_threshold' in solver_info:
            the_zero = solver_info['svd_threshold']
        else:
            the_zero = 1.0e-12

        ind = np.where(lambdas <= the_zero)[0]
        if len(ind) > 0:
            ind = ind[0]
            if ind > 0:
                lambdas = lambdas[:ind]
                lam_len = len(lambdas)
                ratios = np.log(lambdas[:-1]) - np.log(lambdas[1:])
                ind = np.argmax(ratios)
                lambdas_cut = lambdas[:ind]
                U_cut = U[:, :ind]
                V_cut = V[:, :ind]
                D_cut = np.diag(1. / lambdas_cut)
                alpha_vec = np.dot(V_cut, np.dot(D_cut, np.dot(U_cut.T, rhs_vec)))
            else:
                alpha_vec = np.zeros_like(rhs_vec)
        else:
            lam_len = len(lambdas)
            ratios = np.log(lambdas[:-1]) - np.log(lambdas[1:])
            ind = np.argmax(ratios)
            lambdas_cut = lambdas[:ind]
            U_cut = U[:, :ind]
            V_cut = V[:, :ind]
            D_cut = np.diag(1. / lambdas_cut)
            alpha_vec = np.dot(V_cut, np.dot(D_cut, np.dot(U_cut.T, rhs_vec)))
    else:
        raise ValueError('This routine only supports linsolve, mldivide, and truncated svd!!')

    if T == 0:
        opt_val = (np.dot(alpha_vec.T, np.dot(Phi, alpha_vec)) - 2 * np.dot(alpha_vec.T, rhs_vec)) / M
    else:
        opt_val = (np.dot(alpha_vec.T, np.dot(Phi, alpha_vec)) - 2 * np.dot(alpha_vec.T, rhs_vec)) / (T * M)

    fric_coef = None
    return alpha_vec, fric_coef, opt_val
