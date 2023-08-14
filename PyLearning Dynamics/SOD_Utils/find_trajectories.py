import numpy as np
from scipy.integrate import solve_ivp

def find_trajectories(solution, T):
    whole_time = solution.t
    whole_traj = solution.y

    # Find out the first index that the time >= T
    ind = np.where(whole_time >= T)[0][0]

    if whole_time[ind] == T:
        # If T is already included in whole_time, we don't need any interpolation
        traj = whole_traj[:, :ind+1]
        time_vec = whole_time[:ind+1]
    else:
        # If T is not included in whole_time
        # Pre-allocate memory
        traj = np.zeros((whole_traj.shape[0], ind+1))
        time_vec = np.zeros(ind+1)

        # Take those already calculated
        traj[:, :ind] = whole_traj[:, :ind]
        time_vec[:ind] = whole_time[:ind]

        # Now interpolate the solution at T
        t_span = [whole_time[ind-1], whole_time[ind]]
        sol = solve_ivp(solution.fun, t_span, whole_traj[:, ind-1], t_eval=[T])
        traj[:, ind] = sol.y.flatten()
        time_vec[ind] = T

    return traj, time_vec
