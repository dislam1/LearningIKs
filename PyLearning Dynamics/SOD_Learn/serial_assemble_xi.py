import numpy as np
from scipy.sparse import coo_matrix
from pytictoc import TicToc


def serial_assemble_xi(Rs, x, xi, dot_xi, time_vec, sys_info, learn_info):
    VERBOSE = 0

    has_derivative = dot_xi is not None
    M = x.shape[2]
    agents_info = getAgentInfo(sys_info)

    if VERBOSE:
        print('Serial Assemble: Unpacking and initializing data.')

    # Assuming that `tic()` and `toc()` are custom timing functions, let's use `time.time()` instead
    Timings = {}
    t=TicToc()
    Timings['uniformbasisxi'] = t.tic()
    Estimator_Xibasis = uniform_basis_on_xi(Rs, sys_info, learn_info)
    Timings['uniformbasisxi'] = t.toc() - Timings['uniformbasisxi']

    num_xi_Estimator_Phi_cols = np.sum([len(x.f) for row in Estimator_Xibasis for x in row])
    Timings['assembleEstimator_Phi'] = t.tic()

    # Allocate memory for Phi, and rhs_vec, use sparse storage
    Estimator_Phi = coo_matrix((num_xi_Estimator_Phi_cols, num_xi_Estimator_Phi_cols))
    Estimator_rhs_vec = coo_matrix((num_xi_Estimator_Phi_cols, 1))

    # We need the square of the l_2 norm of d_vec - F_vec for all m's
    rhs_in_l2_norm_sq = 0

    # The local Monte Carlo loop
    for m in range(M):
        one_x = x[:, :, m]
        one_xi = xi[:, :, m]

        if has_derivative:
            one_dot_xi = dot_xi[:, :, m]
        else:
            one_dot_xi = None

        one_xi_Estimator_Psi, one_F_vec, one_d_vec, timings = one_step_assemble_on_xi(one_x, one_xi, one_dot_xi, time_vec, agents_info, Estimator_Xibasis, sys_info, learn_info)

        one_rhs_vec = one_d_vec - one_F_vec
        rhs_in_l2_norm_sq += np.linalg.norm(one_rhs_vec) ** 2

        one_PsiT = one_xi_Estimator_Psi.T
        Estimator_Phi += one_PsiT @ one_xi_Estimator_Psi
        Estimator_rhs_vec += one_PsiT @ one_rhs_vec

    extra = {}
    extra['rhs_in_l2_norm_sq'] = rhs_in_l2_norm_sq
    Timings['one_step_assemble_xi'] = timings
    Timings['assembleEstimator_Phi'] = t.toc() - Timings['assembleEstimator_Phi']
    extra['rhoLTXi'] = []

    if VERBOSE:
        print('It takes %10.4e seconds to finish MC assembly.' % Timings['assembleEstimator_Phi'])

    return Estimator_Phi, Estimator_rhs_vec, extra, Timings
