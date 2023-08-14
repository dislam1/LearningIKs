import numpy as np

def get_Riemann_based_time_modifier(N, d, time_vec, Riemann):
    L = len(time_vec)
    time_mod = np.transpose(time_vec[1:L] - time_vec[0:(L - 1)])
    
    if Riemann == 1:
        time_mod = np.concatenate((time_mod, [0]))
    elif Riemann == 2:
        time_mod = np.concatenate(([0], time_mod))
    elif Riemann == 3:
        time_mod = (time_mod + np.concatenate(([0], time_mod))) / 2
    else:
        raise ValueError("Invalid Riemann value")
    
    time_mod = np.sqrt(time_mod)
    time_mod = np.kron(time_mod, np.ones(N * d))
    
    return time_mod
