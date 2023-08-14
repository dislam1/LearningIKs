import numpy as np

def split_traj(dynamics, T):
    # Find out the time instances when the ODE integration is made
    time_vec = dynamics['x']
    # Find out the corresponding trajectory
    traj = dynamics['y']
    # Find out the indices of time_vec in [0, T] and [T, 2T]
    first_indices = np.where(time_vec <= T)[0]
    second_indices = np.where(time_vec >= T)[0]
    # Prepare the storage
    first_traj = traj[:, first_indices]
    first_time = time_vec[first_indices]
    second_traj = traj[:, second_indices]
    second_time = time_vec[second_indices]
    return first_traj, first_time, second_traj, second_time
