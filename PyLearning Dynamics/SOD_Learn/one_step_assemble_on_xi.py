from pytictoc import TicToc


def one_step_assemble_on_xi(x, xi, dot_xi, time_vec, agents_info, xi_basis, sys_info, learn_info):
    Timings = {}
    t=TicToc()
    Timings['approximate_derivative_of_xi'] = t.tic()
    if dot_xi is not None:
        d_vec = dot_xi.flatten()
    else:
        d_vec = approximate_derivative_of_xi(xi, time_vec, sys_info)
    Timings['approximate_derivative_of_xi'] = t.toc() - Timings['approximate_derivative_of_xi']

    Timings['partition_xi'] = t.tic()
    energy_pdist, xi_pdiff, xi_regulator, Psi_row_ind = partition_xi(x, xi, sys_info)
    Timings['partition_xi'] = t.toc() - Timings['partition_xi']

    Timings['assemble_the_learning_matrix_on_xi'] = t.tic()
    Psi = assemble_the_learning_matrix_on_xi(
        energy_pdist, xi_pdiff, xi_regulator, xi_basis, time_vec, agents_info,
        Psi_row_ind, sys_info, learn_info)
    F_vec = find_xi_external(x, xi, sys_info)
    F_vec, d_vec, Psi, _ = scale_the_quantities(
        F_vec, d_vec, Psi, [], sys_info['N'], sys_info['K'], sys_info['type_info'],
        time_vec, learn_info['Riemann_sum'])
    Timings['assemble_the_learning_matrix_on_xi'] = t.toc() - Timings['assemble_the_learning_matrix_on_xi']

    return Psi, F_vec, d_vec, Timings
