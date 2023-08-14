import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def visualize_trajs_for_PNAS(learningOutput, sys_info, chosen_dynamics, obs_info, plot_info):
    trajs = []
    dyn_markers = []
    fine_time_vec = np.linspace(obs_info['time_vec'][0], plot_info['solver_info']['time_span'][1], 100)
    fine_time_vec = np.insert(fine_time_vec, np.searchsorted(fine_time_vec, obs_info['T_L']), obs_info['T_L'])
    coarse_time_vec = np.linspace(fine_time_vec[0], fine_time_vec[-1], 20)
    coarse_time_vec = np.insert(coarse_time_vec, np.searchsorted(coarse_time_vec, obs_info['T_L']), obs_info['T_L'])
    
    for ind in range(len(chosen_dynamics)):
        traj = solve_ivp(chosen_dynamics[ind], [fine_time_vec[0], fine_time_vec[-1]], 
                         chosen_dynamics[ind].y0, t_eval=fine_time_vec)
        trajs.append(traj.y[:sys_info['d'], :])
        
        traj = solve_ivp(chosen_dynamics[ind], [coarse_time_vec[0], coarse_time_vec[-1]], 
                         chosen_dynamics[ind].y0, t_eval=coarse_time_vec)
        dyn_markers.append(traj.y[:sys_info['d'], :])
        
    plot_info['dyn_markers'] = dyn_markers
    plot_info['coarse_time_vec'] = coarse_time_vec
    
    if sys_info['d'] == 1:
        visualize_traj_1D(trajs, fine_time_vec, sys_info, obs_info, plot_info)
    elif sys_info['d'] == 2:
        plot_info['for_PNAS'] = True
        if 'obs_noise' in obs_info and obs_info['obs_noise'] > 0:
            for m_ind in range(learningOutput[0]['obs_data']['ICs'].shape[1]):
                another_y_init = learningOutput[0]['obs_data']['ICs'][:, m_ind]
                if np.linalg.norm(chosen_dynamics[0].y0 - another_y_init, np.inf) == 0:
                    break
            
            traj_noise = learningOutput[0]['obs_data']['x'][:sys_info['d'], :, m_ind]
            visualize_traj_2D_wnoise(traj_noise, trajs[0], trajs[1], fine_time_vec, sys_info, obs_info, plot_info)
        else:
            visualize_traj_2D(trajs, fine_time_vec, sys_info, obs_info, plot_info)
