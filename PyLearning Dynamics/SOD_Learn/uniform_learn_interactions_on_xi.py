import numpy as np
from time import time
from pytictoc import TicToc
import os

def uniform_learn_interactions_on_xi(Rs, x, xi, dot_xi, time_vec, sys_info, learn_info):
    Estimator = {}  # Create an empty dictionary to store the results
    t = TicToc()
    Estimator['Timings']['assemblexi']=t.tic()
    
    if 'VERBOSE' not in learn_info:
        learn_info['VERBOSE'] = 0
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
        
    if not is_parallel:  # Run in serial mode
        Estimator, extra, Estimator['Timings'] = serial_assemble_xi(Rs, x, xi, dot_xi, time_vec, sys_info, learn_info)
        Estimator['Timings']['serial_assemble_xi'] = Estimator['Timings']
    else:
        raise NotImplementedError("Parallel assembly is not yet implemented.")
        # Add the code for parallel_assemble_xi here

    Estimator['Timings']['assemblexi'] = t.toc() - Estimator['Timings']['assemblexi']

    if VERBOSE > 1:
        print('\nIt takes %10.4e seconds to finish the assembly.' % Estimator['Timings']['assemblexi'])
    if VERBOSE > 0:
        print_learning_matrix_info(Estimator['Phi'])

    Estimator['Timings']['solvethelearningproblem'] = t.tic()

    T = time_vec[-1]  # Find out the learning final time
    alpha, opt_val = solve_the_learning_problem_on_xi(Estimator['Phi'], Estimator['rhs_vec'], Estimator['Xibasis'], T, M, learn_info)

    Estimator['alpha'] = alpha
    Estimator['Info']['phiXiSingVals'] = np.linalg.svd(Estimator['Phi'])
    Estimator['Info']['phiXiCond'] = Estimator['Info']['phiXiSingVals'][0] / Estimator['Info']['phiXiSingVals'][-1]

    if VERBOSE > 0:
        print('\nOptimal Value is: % 12.6e.' % opt_val)

    if T == 0:
        Estimator['emp_err'] = opt_val + extra['rhs_in_l2_norm_sq'] / M
    else:
        Estimator['emp_err'] = opt_val + extra['rhs_in_l2_norm_sq'] / (T * M)

    Estimator['Timings']['solvethelearningproblem'] = t.toc() - Estimator['Timings']['solvethelearningproblem']

    if VERBOSE > 0:
        print('\nIt took %10.4e seconds to find the minimizer.' % Estimator['Timings']['solvethelearningproblem'])

    return Estimator, extra
