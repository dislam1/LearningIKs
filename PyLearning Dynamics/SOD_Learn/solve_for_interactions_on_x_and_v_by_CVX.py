import numpy as np
import cvxpy as cp

def solve_for_interactions_on_x_and_v_by_CVX(Phi, rhs_vec, T, M, sys_info, learning_info):
    num_classes = sys_info.K
    CVX_info = learning_info.solver_info.CVX_info
    precision = CVX_info.precision

    has_energy = Phi.shape[1] != 0
    has_align = Phi.shape[1] != 0
    if has_energy and not has_align:
        the_Ms = CVX_info.energy_Ms
        basis_info = [learning_info.Ebasis_info]
        knot_vecs = [learning_info.Ebasis_info.phiEknots]
    elif not has_energy and has_align:
        the_Ms = CVX_info.align_Ms
        basis_info = [learning_info.align_basis_info]
        knot_vecs = [learning_info.align_basis_info.align_knots]
    elif has_energy and has_align:
        the_Ms = np.zeros((num_classes, num_classes, 2))
        the_Ms[:, :, 0] = CVX_info.energy_Ms
        the_Ms[:, :, 1] = CVX_info.align_Ms
        basis_info = [learning_info.Ebasis_info, learning_info.align_basis_info]
        knot_vecs = [learning_info.Ebasis_info.phiEknots, learning_info.align_basis_info.align_knots]
    else:
        raise ValueError('At least one of energy_Phi and align_Phi has to be non-empty!!')

    mod_mats = [[[None for _ in range(num_classes)] for _ in range(num_classes)] for _ in range(len(basis_info))]
    where_to_break = np.zeros_like(the_Ms, dtype=int)
    num_basis = np.zeros_like(the_Ms, dtype=int)
    total_num_basis = np.zeros(len(basis_info), dtype=int)

    for ind in range(len(basis_info)):
        one_basis = basis_info[ind]
        knots = knot_vecs[ind]
        degree = one_basis.degree
        type = one_basis.type

        for k_1 in range(num_classes):
            for k_2 in range(num_classes):
                p = degree[k_1, k_2]
                one_knot = knots[k_1][k_2]

                if type in ['Legendre', 'standard']:
                    if p == 0:
                        mod_mats[ind][k_1][k_2] = polynomial_mod_matrix(one_knot, p, type)
                    elif p == 1:
                        mod_mats[ind][k_1][k_2] = polynomial_mod_matrix(one_knot, p, type)
                    else:
                        raise ValueError('Only piecewise polynomials of degrees 0 or 1 are supported!!')
                else:
                    raise ValueError('Only Legrendre and standard polynomial basis are supported!!')

                n = (p + 1) * (len(one_knot) - 1)
                num_basis[k_1, k_2, ind] = n
                where_to_break[k_1, k_2, ind] = total_num_basis[ind]
                total_num_basis[ind] += n

    if np.sum(total_num_basis) != Phi.shape[1]:
        raise ValueError('total_num_basis does not match up with size(Phi, 2)!!')

    alpha_vec_length = np.sum(total_num_basis)
    alpha_vec = cp.Variable((alpha_vec_length, 1))

    constraints = []

    for ind in range(len(basis_info)):
        if ind > 0:
            prev_sum = total_num_basis[ind - 1]
        else:
            prev_sum = 0

        one_basis = basis_info[ind]
        degree = one_basis.degree

        for k_1 in range(num_classes):
            for k_2 in range(num_classes):
                ind_1 = where_to_break[k_1, k_2, ind] + 1 + prev_sum
                ind_2 = where_to_break[k_1, k_2, ind] + num_basis[k_1, k_2, ind] + prev_sum
                p = degree[k_1, k_2]

                if p == 0:
                    constraints.append(cp.norm(mod_mats[ind][k_1][k_2] @ alpha_vec[ind_1:ind_2], 'inf') <= the_Ms[k_1, k_2, ind])
                elif p == 1:
                    mods = mod_mats[ind][k_1][k_2]
                    mod_1, mod_2 = mods[0], mods[1]
                    constraints.append(cp.norm(mod_1 @ alpha_vec[ind_1:ind_2], 'inf') + cp.norm(mod_2 @ alpha_vec[ind_1:ind_2], 'inf') <= the_Ms[k_1, k_2, ind])

    objective = cp.Minimize(cp.norm(Phi @ alpha_vec - rhs_vec, 'fro'))
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve(solver=CVX_info.solver, verbose=False, abstol=precision, reltol=precision)

    if problem.status != cp.OPTIMAL:
        print(f"The optimization problem did not converge. CVXPY status: {problem.status}")

    energy_alphas = alpha_vec[:total_num_basis[0]].value if has_energy else None
    align_alphas = alpha_vec[total_num_basis[0]:].value if has_align else None
    fric_coef = None  # You need to specify where `fric_coef` is calculated since it's not present in the code.
    return energy_alphas, align_alphas, fric_coef, opt_val
