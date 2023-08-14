import numpy as np

def state_norm(state_vec, sys_info, N_ks, C_ks):
    

    state_vec = state_vec.ravel()
    x = np.array(state_vec[:sys_info['N']* sys_info['d']])  # Extract the component for x from the state vector
    x = x.reshape(sys_info['d'], sys_info['N'])    # Reshape x to its original shape
    the_sum = np.sum(x**2, axis=0)           # Square component-wise and sum over the rows
    state_norm = np.zeros(sys_info['K'])        # Initialize storage for the norm for each class
    for k in range(sys_info['K']):
        state_norm[k] = np.sum(the_sum[C_ks[k, :]]) / N_ks[k]
    state_norm = np.sqrt(np.sum(state_norm))
    return state_norm
