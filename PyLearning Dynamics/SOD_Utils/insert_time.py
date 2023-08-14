import numpy as np
from scipy.integrate import solve_ivp

def insert_time(dynamics, T):
    # Unpack the dynamics result
    time_vec = dynamics.t
    traj = dynamics.y

    # Find the index for the insertion time T
    ind = np.argmax(time_vec >= T)

    # Check if T is already in time_vec
    if time_vec[ind] == T:
        # If T is already in time_vec, split the trajectory accordingly
        first_traj = traj[:, :ind]
        first_time = time_vec[:ind]
        second_traj = traj[:, ind:]
        second_time = time_vec[ind:]
    else:
        # If T is not in time_vec, interpolate the trajectory at T
        traj_at_T = solve_ivp(dynamics.fun, [time_vec[ind - 1], time_vec[ind]], traj[:, ind - 1:ind + 1], t_eval=[T])
        
        # Update the first half and second half of the trajectory
        first_traj = np.concatenate((traj[:, :ind], traj_at_T.y), axis=1)
        first_time = np.concatenate((time_vec[:ind], traj_at_T.t))
        second_traj = np.concatenate((traj_at_T.y, traj[:, ind:]), axis=1)
        second_time = np.concatenate((traj_at_T.t, time_vec[ind:]))

    return first_traj, first_time, second_traj, second_time
