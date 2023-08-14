import numpy as np
from pytictoc import TicToc
from SOD_Utils.calculate_sys_var_len import calculate_sys_var_len
from SOD_Evolve.self_organized_dynamics import self_organized_dynamics
from SOD_Utils.observe_dynamics import observe_dynamics
from SOD_Utils.traj_norm import traj_norm


def computeTrajErr(sys_info, syshat_info, solver_info, obs_info, y_init):
    #Changing values of N
    # Got change somewhere through copy
    #sys_info['N'] = 10
    trajErr = {}
    VERBOSE = obs_info['VERBOSE']
    

    M = y_init.shape[1]
    sys_var_len = calculate_sys_var_len(sys_info)
    traj = np.zeros((sys_var_len, obs_info['L'], M))
    trajfut = np.zeros((sys_var_len, obs_info['L'], M))
    failtraj = np.zeros((M,), dtype=bool)

    # Generate true trajectories
    obs_info_Ltest = obs_info.copy()
    obs_info_Ltest['L'] = obs_info['L']
    obs_info_Ltest['time_vec'] = np.linspace(obs_info['time_vec'][0], obs_info['T_L'], obs_info_Ltest['L'])
    obs_info_Ltest_fut = obs_info_Ltest.copy()
    obs_info_Ltest_fut['time_vec'] = np.linspace(obs_info['T_L'], solver_info['time_span'][1], obs_info_Ltest['L'])
    t=TicToc()
    if VERBOSE > 1:
        print('\nGenerating {} true trajectories for testing...'.format(M))
    trajErr['Timings.TrueKernelTrainICs'] = t.tic()
    for m in range(M):
        dynamics = self_organized_dynamics(y_init[:,m], sys_info, solver_info)
        failtraj[m] = dynamics.flag
        if dynamics.flag:
            continue

        traj[:, :, m] , dtraj,tim_v,elasp_time = observe_dynamics(dynamics, obs_info_Ltest)
        trajfut[:, :, m] , dtraj,tim_v,elasp_time = observe_dynamics(dynamics, obs_info_Ltest_fut)
    trajErr['Timings.TrueKernelTrainICs'] = t.toc(trajErr['Timings.TrueKernelTrainICs'])
    if VERBOSE > 1:
        print('done ({:.2f} secs)'.format(trajErr['Timings.TrueKernelTrainICs']))

    # Generate estimated trajectories
    trajhat = np.zeros((sys_var_len, obs_info['L'], M))
    trajfuthat = np.zeros((sys_var_len, obs_info['L'], M))
    failtrajest = np.zeros((M,), dtype=bool)

    if VERBOSE > 1:
        print('\nGenerating {} estimated trajectories for testing...'.format(M))
    trajErr['Timings.EstKernelTrainICs'] = t.tic()
    for m in range(M):
        dynamics = self_organized_dynamics(y_init[:,m], syshat_info, solver_info)
        failtrajest[m] = dynamics.flag
        if failtrajest[m]:
            continue
        trajhat[:, :, m], dtraj, dvec, dtime = observe_dynamics(dynamics, obs_info_Ltest)
        trajfuthat[:, :, m], dtraj, dvec, dtime = observe_dynamics(dynamics, obs_info_Ltest_fut)
    trajErr['Timings.EstKernelTrainICs'] = t.toc(trajErr['Timings.EstKernelTrainICs'])
    if VERBOSE > 1:
        print('done ({:.2f} sec.)'.format(trajErr['Timings.EstKernelTrainICs']))

    fail_idxs = np.logical_or(failtraj, failtrajest)
    traj[:, :, fail_idxs] = 0
    trajhat[:, :, fail_idxs] = 0
    trajfut[:, :, fail_idxs] = 0
    trajfuthat[:, :, fail_idxs] = 0

    sup_err = np.zeros((M,))
    sup_err_fut = np.zeros((M,))
    if VERBOSE > 1:
        print('\n\tComputing trajectory accuracies...')
    trajErr['Timings.TrajNorm'] = t.tic()
    for m in range(M):
        sup_err[m] = traj_norm(traj[:, :, m], trajhat[:, :, m], 'Time-Maxed', sys_info)
        sup_err_fut[m] = traj_norm(trajfut[:, :, m], trajfuthat[:, :, m], 'Time-Maxed', sys_info)
    trajErr['Timings.TrajNorm'] = t.toc(trajErr['Timings.TrajNorm'])
    if VERBOSE > 1:
        print('done ({:.2f} sec.)'.format(trajErr['Timings.TrajNorm']))

    obs_true = {}
    obs_true['observation'] = traj
    obs_true['observationfuture'] = trajfut
    obs_hat = {}
    obs_hat['observation'] = trajhat
    obs_hat['observationfuture'] = trajfuthat

    trajErr['sup'] = sup_err
    trajErr['sup_fut'] = sup_err_fut
    trajErr['obs_true'] = obs_true
    trajErr['obs_hat'] = obs_hat

    return trajErr
