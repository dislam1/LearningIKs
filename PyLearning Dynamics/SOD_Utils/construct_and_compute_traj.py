import numpy as np
from SOD_Evolve.self_organized_dynamics import self_organized_dynamics
from SOD_Utils.insert_time_vec import insert_time_vec
from SOD_Utils.traj_norm import traj_norm
from SOD_Utils.deval_dynamics import deval_dynamics

def construct_and_compute_traj(solver_info, sys_info, syshat_info, obs_info, ICs):
    result = {}
    for m in range(ICs.shape[1]):
        y_init = ICs[:, m]
        dynamics = self_organized_dynamics(y_init, sys_info, solver_info)
        dynamicshat = self_organized_dynamics(y_init, syshat_info, solver_info)
        if not dynamics.flag and not dynamicshat.flag:
            break

    time_vec = np.arange(obs_info['time_vec'][0], solver_info['time_span'][1], obs_info['time_vec'][1] - obs_info['time_vec'][0])
    time_vec = insert_time_vec(time_vec, obs_info['T_L'])
    traj, dtraj = np.array(deval_dynamics(dynamics, time_vec))
    trajhat, dtrajhat = np.array(deval_dynamics(dynamicshat, time_vec))

    result['traj_true'] = traj[:sys_info['d'] * sys_info['N'], :]
    result['traj_hat'] = trajhat[:sys_info['d'] * sys_info['N'], :]
    result['time_vec'] = time_vec

    # Compute the trajectory error from observed time instances
    time_vec = obs_info['time_vec']
    traj, dtraj = deval_dynamics(dynamics, time_vec)
    trajhat, dtrajhat = deval_dynamics(dynamicshat, time_vec)
    result['trajErr'] = traj_norm(traj, trajhat, 'Time-Maxed', sys_info)

    # Compare the prediction time interval [T_L, T_f]
    time_vec = np.arange(obs_info['T_L'], solver_info['time_span'][1], obs_info['time_vec'][1] - obs_info['time_vec'][0])
    traj, dtraj = np.array(deval_dynamics(dynamics, time_vec))
    trajhat, dtrajhat = np.array(deval_dynamics(dynamicshat, time_vec))
    result['trajErrfut'] = traj_norm(traj, trajhat, 'Time-Maxed', sys_info)

    result['m'] = m
    result['dynamics'] = dynamics
    result['dynamicshat'] = dynamicshat

    return result
