import numpy as np
from pytictoc import TicToc
from SOD_Learn.learn_interactions_on_x_and_v import learn_interactions_on_x_and_v
from SOD_Learn.learn_interactions_on_xi import learn_interactions_on_xi


def learn_from_dynamics(sys_info, obs_info, learn_info, obs_data):
    learn_out = {}
    learn_out['Timings'] = {}
    t=TicToc()
    learn_out['Timings']['total'] =t.tic()
    
    one_block = sys_info['d'] * sys_info['N']
    
    if 'VERBOSE' not in learn_info:
        learn_info['VERBOSE'] = 2
    VERBOSE = learn_info['VERBOSE']
    #print('\n\t Learn from dynamics - obs data')
    #print(obs_data)
    if obs_data is None or len(obs_data['x']) == 0:
        print('\n\t WARNING: empty observations!')
        return learn_out
    
    x = obs_data['x'][0:one_block, :, :]
    
    if sys_info['ode_order'] == 1:
        v = None
        xi = None
        Estimator_xi = None
        extra_xi = None
        sys_info['has_xi'] = False
        
        if obs_info['use_derivative']:
            dot_xv = obs_data['xp'][0:one_block, :, :]
        else:
            dot_xv = None
    elif sys_info['ode_order'] == 2:
        v = obs_data['x'][one_block:2 * one_block, :, :]
        
        if obs_info['use_derivative']:
            dot_xv = obs_data['xp'][one_block:2 * one_block, :, :]
        else:
            dot_xv = None
        
        if sys_info['has_xi']:
            xi = obs_data['x'][2 * one_block:2 * one_block + sys_info['N'], :, :]
            
            if obs_info['use_derivative']:
                dot_xi = obs_data['xp'][2 * one_block:2 * one_block + sys_info['N'], :, :]
            else:
                dot_xi = None
        else:
            xi = None
            dot_xi = None
            Estimator_xi = None
            extra_xi = None
    
    if VERBOSE > 1:
        print('\n===============================================')
        print('Learning interaction kernel for x and v.')
    
    learn_out['Timings']['learnInteractions'] = t.tic()
    Estimator, extra_xv = learn_interactions_on_x_and_v(x, v, xi, dot_xv, obs_info['time_vec'], sys_info, learn_info)
    learn_out['Timings']['learnInteractions'] = t.toc(learn_out['Timings']['learnInteractions'])
    learn_out['Timings']['learn_interactions_on_x_and_v'] = Estimator['Timings']
    
    if sys_info['has_xi']:
        if VERBOSE > 1:
            print('\n===============================================')
            print('Done learning interaction kernel for x and v, starting learning on xi.')
        
        Rs = extra_xv['Rs']
        learn_out['Timings']['learnInteractionsXi'] = t.tic()
        Estimator_xi, extra_xi = learn_interactions_on_xi(Rs, x, xi, dot_xi, obs_info['time_vec'], sys_info, learn_info)
        learn_out['Timings']['learnInteractionsXi'] = t.toc(learn_out['Timings']['learnInteractionsXi'])
    
    if not learn_info['MEMORY_LEAN']:
        learn_out['Phi'] = Estimator['Phi']
        learn_out['rhs'] = Estimator['rhs']
        
        if Estimator_xi is not None:
            learn_out['PhiXi'] = Estimator_xi['PhiXi']
            learn_out['rhsXi'] = Estimator_xi['rhsXi']
    
    learn_out['extra_xv'] = extra_xv
    rhoLTemp = extra_xv['rhoLTemp']
    
    if Estimator_xi is not None:
        Estimator['phiXihat'] = Estimator_xi['phiXihat']
        Estimator['Xibasis'] = Estimator_xi['Xibasis']
        Estimator['emp_err_xi'] = Estimator_xi['emp_err']
        learn_out['extra_xi'] = extra_xi
        rhoLTemp['rhoLTXi'] = extra_xi['rhoLTXi']
    else:
        Estimator['phiXihat'] = None
        Estimator['Xibasis'] = None
        Estimator['emp_err_xi'] = None
        learn_out['extra_xi'] = None
        #rhoLTemp['rhoLTXi'] = None
    
    learn_out['rhoLTemp'] = rhoLTemp
    learn_out['Timings']['total'] = t.toc(learn_out['Timings']['total'])
    learn_out['Estimator'] = Estimator
    
    return learn_out
