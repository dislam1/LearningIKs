import numpy as np

def partition_sys_var(y, sys_info):
    # Check if y is a column vector
    
    if y.ndim != 1:
        raise ValueError("y has to be a column vector!")
    
    block_size = sys_info['d'] * sys_info['N']
    #ynp.array(y).reshape(block_size,1)
   
    #x = np.zeros(y[1:block_size], sys_info['d'], sys_info['N'])
    
    if sys_info['ode_order'] == 1:
        v = None
        xi = None
        x = np.reshape(y[:block_size], (sys_info['d'], sys_info['N']))
    elif sys_info['ode_order'] == 2:
        v = np.reshape(y[block_size:2 * block_size], (sys_info['d'], sys_info['N']))
        if sys_info['has_xi']:
            xi = np.reshape(y[2 * block_size:2 * block_size + sys_info['N']], (1, sys_info['N']))
        else:
            xi = None
    
    state_vars = {'x': x, 'v': v, 'xi': xi}
    return state_vars

# Assuming sys_info and y are available as inputs to this function.

# Example usage:
# state_vars = partition_sys_var(y, sys_info)
