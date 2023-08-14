import numpy as np
from scipy.interpolate import interp1d
from pytictoc import TicToc

def deval_dynamics(solution, obs_vec):
   

    traj, dtraj = None, None
    try:
        # Find out the trajectory position and derivative at the specified time instances
        # Calculate the derivative solution.yp using NumPy's gradient function
        solution.yp = np.gradient(solution.x, solution.tv, axis=1)
        #traj_interp = interp1d(solution.t, (solution.y).T, kind='cubic', axis=0)
        #dtraj_interp = interp1d(solution.t, solution.sol(solution.t)[1].T, kind='cubic', axis=0)
        

        traj_interp = interp1d(solution.tv, solution.x, axis=1, fill_value="extrapolate", bounds_error=False)  # Interpolate the trajectory
        dtraj_interp = interp1d(solution.tv, solution.yp, axis=1, fill_value="extrapolate", bounds_error=False)  # Interpolate the derivative

        traj = np.array(traj_interp(obs_vec))
        dtraj = np.array(dtraj_interp(obs_vec))
        #Af = [list(i) for i in traj]
        #[i.sort(reverse=True) for i in Af]
        #traj=np.array(Af)

        #Bf = [list(i) for i in dtraj]
        #[i.sort(reverse=True) for i in Bf]
        #dtraj=np.array(Bf)

      
    except Exception:
        traj = np.NaN
        dtraj = np.NaN

    return traj, dtraj

# Usage example:
# Assuming you have the solution of the ODE stored in 'solution' and 'obs_info' dictionary available
# traj, dtraj, time_vec, timings = observe_dynamics(solution, obs_info)
