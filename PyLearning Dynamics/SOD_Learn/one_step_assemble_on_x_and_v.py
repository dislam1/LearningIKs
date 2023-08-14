from pytictoc import TicToc
from SOD_Learn.approximate_derivative_of_x_or_v import approximate_derivative_of_x_or_v
from SOD_Learn.partition_x_and_v import partition_x_and_v
from SOD_Learn.find_x_or_v_external import find_x_or_v_external
from SOD_Learn.assemble_the_learning_matrix_on_x_and_v import assemble_the_learning_matrix_on_x_and_v
from SOD_Learn.scale_the_quantities import scale_the_quantities

def one_step_assemble_on_x_and_v(x, v, xi, dot_xv, time_vec, agents_info, energy_basis, align_basis, sys_info, learn_info):
    Timings = {}
    t=TicToc()
    Timings['approximate_derivative_of_x_or_v'] = t.tic()
    if dot_xv is not None:
        d_vec = dot_xv.flatten()
    else:
        d_vec = approximate_derivative_of_x_or_v(x, v, time_vec, sys_info)
    Timings['approximate_derivative_of_x_or_v'] = t.toc(Timings['approximate_derivative_of_x_or_v'])

    Timings['partition_x_and_v'] = t.tic()
    energy_pdist, energy_pdiff, energy_reg, _, align_pdiff, align_reg, Psi_row_ind = partition_x_and_v(x, v, sys_info, learn_info)
    Timings['partition_x_and_v'] = t.toc( Timings['partition_x_and_v'])

    Timings['assemble_the_learning_matrix_on_x_and_v'] = t.tic()
    F_vec = find_x_or_v_external(x, v, xi, sys_info)
    energy_Psi, align_Psi = assemble_the_learning_matrix_on_x_and_v(
        energy_pdist, energy_pdiff, energy_reg, energy_basis, align_pdiff, align_reg, align_basis,
        time_vec, agents_info, Psi_row_ind, sys_info, learn_info)
    F_vec, d_vec, energy_Psi, align_Psi = scale_the_quantities(
        F_vec, d_vec, energy_Psi, align_Psi, sys_info['N'], sys_info['K'],
        sys_info['type_info'], time_vec, learn_info['Riemann_sum'])
    Timings['assemble_the_learning_matrix_on_x_and_v'] = t.toc(Timings['assemble_the_learning_matrix_on_x_and_v'])

    return energy_Psi, align_Psi, F_vec, d_vec, Timings
