import numpy as np

def generatePerturbedTrajectories(sys_info, solver_info, y_init, obs_info, traj, obs_info_fut, traj_fut, SIGMA, N_PERTURBATIONS):
    failtrajpert = np.zeros(N_PERTURBATIONS, dtype=bool)
    
    flag_fut = (obs_info_fut is not None) and (traj_fut is not None)
    
    y_init_pert = y_init + SIGMA * np.sqrt(sys_info['d']) / np.linalg.norm(y_init, axis=0) * np.random.randn(*y_init.shape)
    
    supdist = np.zeros(N_PERTURBATIONS)
    supdist_fut = np.zeros(N_PERTURBATIONS)
    
    for k in range(N_PERTURBATIONS):
        dynamics = self_organized_dynamics(y_init_pert[:, k], sys_info, solver_info)
        failtrajpert[k] = dynamics.flag
        if failtrajpert[k]:
            continue
        
        trajpert = observe_dynamics(dynamics, obs_info)
        supdist[k] = traj_norm(traj, trajpert, 'Time-Maxed', sys_info)
        
        if flag_fut:
            trajpert_fut = observe_dynamics(dynamics, obs_info_fut)
            supdist_fut[k] = traj_norm(traj_fut, trajpert_fut, 'Time-Maxed', sys_info)
    
    sup_pert = np.mean(supdist[~failtrajpert])
    
    if flag_fut:
        sup_pert_fut = np.mean(supdist_fut[~failtrajpert])
    else:
        sup_pert_fut = None
        
    return sup_pert, sup_pert_fut, failtrajpert
