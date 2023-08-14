import numpy as np
import os
from time import time
from pytictoc import TicToc
from SOD_Learn.serial_assemble_x_and_v import serial_assemble_x_and_v
from SOD_Utils.print_learning_matrix_info import print_learning_matrix_info
from SOD_Learn.solve_the_learning_problem_on_x_and_v import solve_the_learning_problem_on_x_and_v


def uniform_learn_interactions_on_x_and_v(x, v, xi, dot_xv, time_vec, sys_info, learn_info):
    Estimator = {}  # Create an empty dictionary to store the results
    Estimator['Timings']={}
    t = TicToc()
    Estimator['Timings']['serialassemblexandv'] = t.tic()
    
    if 'VERBOSE' not in learn_info:
        learn_info['VERBOSE'] = 2
    VERBOSE = learn_info['VERBOSE']

    M = x.shape[2]  # Find out the total number of MC realizations

    if 'is_parallel' in learn_info:
        is_parallel = learn_info['is_parallel']
    else:
        is_parallel = False

   # if M < os.cpu_count():  # Replace 'os.cpu_count()' with the actual number of CPU cores available
    is_parallel = False

    if VERBOSE > 1:
        print('\nAssembling Phi, rhs_vec and other related quantities.')
        
    if not is_parallel:  # Run in serial mode
        Estimator, extra, Estimator['Timings'] = serial_assemble_x_and_v(x, v, xi, dot_xv, time_vec, sys_info, learn_info)
        Estimator['Timings']['serial_assemble_x_and_v'] = Estimator['Timings']
    else:
        raise NotImplementedError("Parallel assembly is not yet implemented.")
        # Add the code for parallel_assemble here

    #Estimator['Timings']['serialassemblexandv'] = t.toc(Estimator['Timings']['serialassemblexandv'])
    #Estimator['Timings']['serialassemblexandv'] = t.toc(Estimator['Timings']['serialassemblexandv'])

    if VERBOSE > 1:
        print('\nIt takes %10.4e seconds to finish the assembly.' % Estimator['Timings']['serialassemblexandv'])
    if VERBOSE > 1:
        print('\nPrinting information about learning matrix:')
        print_learning_matrix_info(Estimator['Phi'])
    if VERBOSE > 1 and 'Phi_xi' in Estimator:
        print('\nPrinting information about learning matrix on xi:')
        print_learning_matrix_info(Estimator['Phi_xi'])

    Estimator['Timings']['solvethelearningproblem'] = t.tic()  # Construct the learning problem on x and v

    T = time_vec[-1]  # Find out the learning final time
    alpha, _, opt_val = solve_the_learning_problem_on_x_and_v(Estimator['Phi'], Estimator['rhs_vec'], T, M, sys_info, learn_info)
    #Add Info in Estimator
    Estimator['Info'] = {}

    Estimator['alpha'] = alpha
    Estimator['Info']['phiSingVals'] = np.linalg.svd(Estimator['Phi'])
    
    Estimator['Info']['phiCond'] = Estimator['Info']['phiSingVals'][0] / Estimator['Info']['phiSingVals'][-1]

    if VERBOSE > 1:
        print('\nOptimal Value on x/v is: % 12.6e.' % opt_val)
    if VERBOSE > 1 and 'Phi_xi' in Estimator:
        print('\nOptimal Value on xi is: % 12.6e.' % opt_val_xi)

    if T == 0:
        Estimator['emp_err'] = opt_val + extra['rhs_in_l2_norm_sq'] / M
    else:
        Estimator['emp_err'] = opt_val + extra['rhs_in_l2_norm_sq'] / (T * M)

    Estimator['Timings']['solvethelearningproblem'] = t.toc(Estimator['Timings']['solvethelearningproblem'])

    if VERBOSE > 1:
        print('\nIt took %10.4e seconds to find the minimizer.' % Estimator['Timings']['solvethelearningproblem'])

    return Estimator, extra
