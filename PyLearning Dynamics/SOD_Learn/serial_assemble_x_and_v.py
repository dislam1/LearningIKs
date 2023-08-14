import numpy as np
from scipy.sparse import coo_matrix
from pytictoc import TicToc
from SOD_Utils.getAgentInfo import getAgentInfo
from SOD_Learn.find_maximums import find_maximums
from SOD_Learn.uniform_basis_on_x_and_v import uniform_basis_on_x_and_v
from SOD_Learn.one_step_assemble_on_x_and_v import one_step_assemble_on_x_and_v


def serial_assemble_x_and_v(x, v, xi, dot_xv, time_vec, sys_info, learn_info):
    VERBOSE = 0
    M = x.shape[2]
    has_v = v is not None
    has_xi = xi is not None
    has_derivative = dot_xv is not None
    has_align = v is not None and learn_info['Abasis_info'] is not None
    has_energy = learn_info['Ebasis_info'] is not None
    agents_info = getAgentInfo(sys_info)
    N_per_class = np.histogram(sys_info['type_info'], bins=np.arange(sys_info['K'] + 2))[0]
    
    Timings = {}
    t=TicToc()
    Timings['findRs'] = t.tic()
    #Concatenate x, v and xi
    xv = np.concatenate((x, np.broadcast_to(np.array(v)[None, None, None], x.shape[:-1] + (1,))), axis = -1)
    Mtrajs = np.concatenate((xv, np.broadcast_to(np.array(xi)[None, None, None], xv.shape[:-1] + (1,))), axis = -1)
    #if (np.array(v)).size == 0:
    #
    # #else:
    #    Mtrajs =np.hstack((x, v))
    #if (np.array(xi)).size > 0:
    #    Mtrajs =np.hstack((Mtrajs, xi))

    max_rs = np.zeros((sys_info['K'], sys_info['K'], M))
    for m in range(M):
        traj =np.squeeze(Mtrajs[:, :, m])
        output = find_maximums(traj, sys_info)
        max_rs[:, :, m] = output['max_rs']
    Rs = np.max(max_rs, axis=2)
    Timings['findRs'] = t.toc(Timings['findRs'])
    
    if VERBOSE > 1:
        print('\n\tAssembling Rs takes %10.4e sec.' % Timings['findRs'])
    
    if VERBOSE > 1:
        print('Construct the basis...')
    Timings['uniformbasisxv'] = t.tic()
    Estimator = {}
    Estimator['Ebasis'], Estimator['Abasis'] = uniform_basis_on_x_and_v(Rs, sys_info, learn_info)
    Timings['uniformbasisxv'] = t.toc(Timings['uniformbasisxv'])
    
    num_energy_Estimator = {}
    if has_energy:
        num_energy_Estimator['Phi_cols'] = np.sum([len(l['f']) for i, j in enumerate(Estimator['Ebasis']) for k, l in enumerate(j)])
    else:
        num_energy_Estimator['Phi_cols'] =  0
    
    num_align_Estimator = {}
    if has_align:
        num_align_Estimator['Phi_cols'] = np.sum([len(l['f']) for i, j in enumerate(Estimator['Abasis']) for k, l in enumerate(j)])
    else:
        num_align_Estimator['Phi_cols'] = 0
    
    if VERBOSE > 1:
        print('done (%10.4e sec).' % Timings['uniformbasisxv'])
    Timings['assembleEstimator'] ={}
    Timings['assembleEstimator']['Phi'] = t.tic()
    num_Estimator = {}
    num_Estimator['Phi_cols'] = num_energy_Estimator['Phi_cols'] + num_align_Estimator['Phi_cols']

    #Allocate memmory and define Estimator structure
    
    #Estimator['Phi'] = coo_matrix((num_Estimator['Phi_cols'], num_Estimator['Phi_cols']))
    Estimator['Phi'] = np.zeros((num_Estimator['Phi_cols'], num_Estimator['Phi_cols']))
    Estimator['rhs_vec'] = np.zeros((num_Estimator['Phi_cols'],1))
    rhs_in_l2_norm_sq = 0
    if VERBOSE>1:
        print('\nAssembling the matrices for the optimization problem...')
    for m in range(M):
        one_x = x[:, :, m]
        one_v = v[:, :, m] if has_v else None
        one_xi = xi[:, :, m] if has_xi else None
        one_dot_xv = dot_xv[:, :, m] if has_derivative else None
        
        one_energy_Estimator={}
        one_align_Estimator={}
        one_energy_Estimator['Psi'], one_align_Estimator['Psi'], one_F_vec, one_d_vec, timings = \
            one_step_assemble_on_x_and_v(one_x, one_v, one_xi, one_dot_xv, time_vec, agents_info, 
                                         Estimator['Ebasis'], Estimator['Abasis'], sys_info, learn_info)
        
        one_rhs_vec = np.array(one_d_vec - one_F_vec)
        one_rhs_vec = one_rhs_vec.reshape(one_rhs_vec.shape[0],-1)
        if (np.array(one_align_Estimator['Psi'])).size == 0:
            one_Psi = one_energy_Estimator['Psi']
        else:
            one_Psi = np.hstack((one_energy_Estimator['Psi'], one_align_Estimator['Psi']))
        rhs_in_l2_norm_sq += np.linalg.norm(one_rhs_vec) ** 2
        
        one_PsiT = one_Psi.T
        Estimator['Phi'] += one_PsiT @ one_Psi
        Estimator['rhs_vec'] += one_PsiT @ one_rhs_vec
    
    Timings['one_step_assemble_on_x_and_v'] = timings
    Timings['assembleEstimator']['Phi'] = t.toc(Timings['assembleEstimator']['Phi'])
    
    if VERBOSE > 1:
        print('It took %10.4e seconds to assemble Estimator.Phi and Estimator.rhs_vec.' % Timings['assembleEstimator.Phi'])
    
    extra = {}
    if has_xi:
        extra['Rs'] = Rs
    extra['rhs_in_l2_norm_sq'] = rhs_in_l2_norm_sq
    extra['rhoLTemp'] = None
    
    return Estimator, extra, Timings
