import  pickle

from SOD_Utils.visualize_traj_1D import visualize_traj_1D
from SOD_Utils.visualize_traj_2D import visualize_traj_2D
from SOD_Utils.visualize_traj_3D import visualize_traj_3D
from SOD_Utils.construct_and_compute_traj import construct_and_compute_traj
from SOD_Utils.generateICs import generateICs
from SOD_Utils. visualize_traj_2D_wnoise import  visualize_traj_2D_wnoise


def visualizetrajs(learningOutput, sys_info, sys_info_approx, obs_info, ICs, plot_info):
    trajs = [None] * 4
    result = construct_and_compute_traj(plot_info['solver_info'], sys_info, sys_info_approx, obs_info, ICs)
    trajs[0] = result['traj_true']
    trajs[1] = result['traj_hat'] 
    time_vec = result['time_vec']

    if sys_info['d'] == 2:
        if sys_info['name'] in ['PredatorPrey1stOrder', 'PredatorPrey1stOrderSplines', 'PredatorPrey2ndOrder']:
            if 'obs_noise' in obs_info and obs_info['obs_noise'] > 0 and not plot_info['for_larger_N']:
                traj_noise = learningOutput[0]['obs_data']['x'][:sys_info['d'] * sys_info['N'], :, result['m']]
                visualize_traj_2D_wnoise(traj_noise, result['traj_true'], result['traj_hat'], time_vec, sys_info, obs_info, plot_info)

    print('\n================================================================================')
    print('\n------------------- Trajectory Error over One Initial Condition')
    
    if 'for_larger_N' not in plot_info or not plot_info['for_larger_N']:
        print('\nTraj. Err. with an initial condition taken from training data:')
        chosen_dynamics = [None] * 4
        chosen_dynamics[0] = result['dynamics']
        chosen_dynamics[1] = result['dynamicshat']
    else:
        chosen_dynamics_LN = [None] * 4
        chosen_dynamics_LN[0] = result['dynamics']
        chosen_dynamics_LN[1] = result['dynamicshat']
        print(' For Larger N')
        print('\nTraj. Err. with an initial condition randomly chosen:')
    
    print(f"\n  sup-norm on [{obs_info['time_vec'][0]:.4e},{obs_info['T_L']:.4e}] = {result['trajErr']:.4e}.")
    print(f"  sup-norm on [{obs_info['T_L']:.4e},{plot_info['solver_info']['time_span'][1]:.4e}] = {result['trajErrfut']:.4e}.")
    
    # Randomly pick another initial data
    if plot_info['for_larger_N']:
        ICs = ICs[:, result['m']:]
    else:
        if 'y_init_new' in learningOutput[0]:
            ICs = learningOutput[0]['y_init_new']
        else:
            ICs = generateICs(obs_info['M'], sys_info)
            
    result = construct_and_compute_traj(plot_info['solver_info'], sys_info, sys_info_approx, obs_info, ICs)
    trajs[2] = result['traj_true']
    trajs[3] = result['traj_hat']

    if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
        chosen_dynamics_LN[2] = result['dynamics']
        chosen_dynamics_LN[3] = result['dynamicshat']
        print('  For Larger N')
    else:
        chosen_dynamics[2] = result['dynamics']
        chosen_dynamics[3] = result['dynamicshat']

    print('\nTraj. Err. with another initial condition randomly chosen:')
    print(f"\n  sup-norm on [{obs_info['time_vec'][0]:.4e},{obs_info['T_L']:.4e}] = {result['trajErr']:.4e}.")
    print(f"  sup-norm on [{obs_info['T_L']:.4e},{plot_info['solver_info']['time_span'][1]:.4e}] = {result['trajErrfut']:.4e}.")

    # Put the trajectories on one single window for comparison
    switch = {
        1: visualize_traj_1D,
        2: visualize_traj_2D,
        3: visualize_traj_3D,
    }
    switch[sys_info['d']](trajs, time_vec, sys_info, obs_info, plot_info)

  # Save the dynamics
    if 'save_file' in plot_info and plot_info['save_file']:
        if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
            with open(plot_info['save_file'], 'ab') as f:
                pickle.dump(chosen_dynamics_LN, f)
        else:
            with open(plot_info['save_file'], 'ab') as f:
                pickle.dump(chosen_dynamics, f)