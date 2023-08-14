import time
import copy
import numpy as np
from pytictoc import TicToc

def LearnAndComputeTrajectoryErr(obs_info, solver_info, learn_info, obs_data):
    # Learn an estimator and regularize it
    t = TicToc()
    t.tic()
    
    if learn_info['sys_info']['ode_order'] == 1:
        ode_sym = '1st order'
    elif learn_info['sys_info']['ode_order'] == 2:
        ode_sym = '2nd order'
    
    print(f'\nLearning Interactions as if {ode_sym} system......')
    
    learn_out = learn_from_dynamics(learn_info['sys_info'], obs_info, learn_info, obs_data)
    learn_out['rhoLTemp'] = estimateRhoLT(obs_data, learn_info['sys_info'], obs_info)
    
    phiEhatsmooth, Ebasis2 = regularizeInfluenceFunction(learn_out['Estimator']['phiEhat'],
                                                         learn_out['Estimator']['Ebasis'],
                                                         learn_out['rhoLTemp']['rhoLTE'],
                                                         learn_info['sys_info'])
    
    learn_out['Estimator']['phiEhatsmooth'] = phiEhatsmooth
    learn_out['Estimator']['Ebasis2'] = Ebasis2
    
    t.toc()
    print(f'done ({t.elapsed:.2f} secs).')
    print(f'The empirical error from learning phis on x/v is: {learn_out["Estimator"]["emp_err"]:10.4e}.')
    
    print('Computing trajectory errors...')
    
    t.tic()
    syshatsmooth_info = copy.deepcopy(learn_info['sys_info'])
    syshatsmooth_info['phiE'] = learn_out['Estimator']['phiEhatsmooth']
    traj = obs_data['x']
    M = traj.shape[2]
    trajhat = np.zeros(traj.shape)
    failtrajest = np.zeros(M, dtype=bool)
    
    for m in range(M):
        dynamicshat = self_organized_dynamics(obs_data['ICs'][:, m], syshatsmooth_info, solver_info)
        failtrajest[m] = dynamicshat['flag']
        if not failtrajest[m]:
            trajhat[:, :, m] = observe_dynamics(dynamicshat, obs_info)
    
    if np.sum(failtrajest) > 0:
        print(f'\n\t Failed simulations with estimated system: {np.sum(failtrajest)}/{M}')
    
    traj = traj[:, :, ~failtrajest]
    trajhat = trajhat[:, :, ~failtrajest]
    M = traj.shape[2]
    trajErr = np.zeros(M)
    
    for m in range(M):
        trajErr[m] = traj_norm(traj[:, :, m], trajhat[:, :, m], 'Time-Maxed', learn_info['sys_info'])
    
    t.toc()
    print(f'done ({t.elapsed:.2f} secs).')
    print(f'------------------- Trajectory accuracies, same IC''s as training data, ({ode_sym}):')
    print(f'\tsup-norm  on [0,{obs_info["T_L"]}] = {np.mean(trajErr):10.4e} ± {np.std(trajErr):10.4e}')

    # Package the data
    output = {
        'learn_out': learn_out,
        'trajErr': trajErr
    }
    return output
