import numpy as np
from scipy.interpolate import interp1d
from pytictoc import TicToc

def observe_dynamics(solution, obs_info):
    Timings = TicToc()
    Timings.tic()
    #print('\n time vec...')
    #print(obs_info['time_vec'])

    if 'time_vec' not in obs_info or obs_info['time_vec'].size == 0:
        if obs_info['L'] == 1:
            obs_info['time_vec'] = [0]  # When we only want one observation, we take the initial time
        else:
            obs_info['time_vec'] = np.linspace(0, obs_info['T_L'], obs_info['L'])  # Default equi-distance times
    elif len(obs_info['time_vec']) != obs_info['L']:
        print('\n\tWarning: observe_dynamics: Length of obs_info.time_vec != L. Using time_vec.')

    traj, dtraj = None, None
    try:
        # Find out the trajectory position and derivative at the specified time instances
        # Calculate the derivative solution.yp using NumPy's gradient function
        solution.yp = np.gradient(solution.x, solution.tv, axis=1)
        #traj_interp = interp1d(solution.t, (solution.y).T, kind='cubic', axis=0)
        #dtraj_interp = interp1d(solution.t, solution.sol(solution.t)[1].T, kind='cubic', axis=0)
        

        traj_interp = interp1d(solution.tv, solution.x, axis=1, fill_value="extrapolate", bounds_error=False)  # Interpolate the trajectory
        dtraj_interp = interp1d(solution.tv, solution.yp, axis=1, fill_value="extrapolate", bounds_error=False)  # Interpolate the derivative

        traj = np.array(traj_interp(obs_info['time_vec']))
        dtraj = np.array(dtraj_interp(obs_info['time_vec']))

        #Af = [list(i) for i in traj]
        #[i.sort(key=abs) for i in Af]
        #traj=np.array(Af)

        #Bf = [list(i) for i in dtraj]
        #[i.sort(key=abs) for i in Bf]
        #dtraj=np.array(Bf)


        if 'obs_noise' in obs_info and obs_info['obs_noise'] > 0:
            traj += obs_info['mu_trajnoise'](traj, obs_info['obs_noise'])
            dtraj += obs_info['mu_dtrajnoise'](dtraj, obs_info['obs_noise'])
    except Exception:
        traj = np.NaN
        dtraj = np.NaN

    time_vec = obs_info['time_vec']
    Timings.toc('Obtained trajectory and derivative in ')
    

    return traj, dtraj, time_vec, Timings.elapsed

# Usage example:
# Assuming you have the solution of the ODE stored in 'solution' and 'obs_info' dictionary available
# traj, dtraj, time_vec, timings = observe_dynamics(solution, obs_info)
