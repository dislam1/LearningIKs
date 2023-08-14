import numpy as np
from pytictoc import TicToc
import os



def uniform_learn_interactions(x, v, xi, dot_xv, dot_xi, time_vec, sys_info, learn_info):
    Estimator = {}  # Create an empty dictionary to store the results
    t=TicToc()
    if 'VERBOSE' not in learn_info:
        learn_info['VERBOSE'] = 2
    VERBOSE = learn_info['VERBOSE']
    M = x.shape[2]  # Find out the total number of MC realizations

    if 'is_parallel' in learn_info:
        is_parallel = learn_info['is_parallel']
    else:
        is_parallel = False
    if M < os.cpu_count():  # Replace 'os.cpu_count()' with the actual number of CPU cores available
        is_parallel = False

    if VERBOSE > 1:
        print('\nAssembling Phi, rhs_vec and other related quantities.')
    if not is_parallel:
        Estimator = serial_assemble(x, v, xi, dot_xv, dot_xi, time_vec, sys_info, learn_info)
    else:
        raise NotImplementedError("Parallel assembly is not yet implemented.")
        # Add the code for parallel_assemble here

    if VERBOSE > 1:
        print('\nIt takes %10.4e seconds to finish the assembly.' % Estimator['Timings']['assemble'])
    if VERBOSE > 1:
        print('\nPrinting information about learning matrix on x/v:')
        print_learning_matrix_info(Estimator['Phi'])
    if VERBOSE > 1 and 'Phi_xi' in Estimator:
        print('\nPrinting information about learning matrix on xi:')
        print_learning_matrix_info(Estimator['Phi_xi'])

    Estimator['Timings']['solve'] = t.tic()  # Construct the learning problem on x and v
    T = time_vec[-1]  # Find out the learning final time
    alpha, alpha_xi, opt_val, opt_val_xi = solve_the_learning_problem(Estimator, T, M, learn_info)  # Find out the minimizer and optimal value
    Estimator['alpha'] = alpha
    Estimator['alpha_xi'] = alpha_xi
    Estimator['Info']['phiSingVals'] = np.linalg.svd(Estimator['Phi'])
    Estimator['Info']['phiCond'] = Estimator['Info']['phiSingVals'][0] / Estimator['Info']['phiSingVals'][-1]
    if 'Phi_xi' in Estimator:
        Estimator['Info']['phiSingVals_xi'] = np.linalg.svd(Estimator['Phi_xi'])
        Estimator['Info']['phiCond_xi'] = Estimator['Info']['phiSingVals_xi'][0] / Estimator['Info']['phiSingVals_xi'][-1]

    if VERBOSE > 1:
        print('\nOptimal Value on x/v is: % 12.6e.' % opt_val)
    if VERBOSE > 1 and 'Phi_xi' in Estimator:
        print('\nOptimal Value on xi is: % 12.6e.' % opt_val_xi)

    if T == 0:
        Estimator['emp_err'] = opt_val + Estimator['rhs_in_l2_norm_sq'] / M
        if 'Phi_xi' in Estimator:
            Estimator['emp_err_xi'] = opt_val_xi + Estimator['rhs_in_l2_norm_sq'] / M
    else:
        Estimator['emp_err'] = opt_val + Estimator['rhs_in_l2_norm_sq'] / (T * M)
        if 'Phi_xi' in Estimator:
            Estimator['emp_err_xi'] = opt_val_xi + Estimator['rhs_in_l2_norm_sq'] / (T * M)

    Estimator['Timings']['solve'] = t.toc() - Estimator['Timings']['solve']
    if VERBOSE > 1:
        print('\nIt took %10.4e seconds to find the minimizer.' % Estimator['Timings']['solve'])

    return Estimator
