import numpy as np

def find_x_or_v_external(x, v, xi, sys_info):
    d = sys_info['d']
    N = sys_info['N']
    if v is None:
        L = 0
    else:
        L = v.shape[1]
    ode_order = sys_info['ode_order']
    external = np.zeros(x.shape)
    
    if ode_order == 2:
        v_external = sys_info['Fv']
    
    for l in range(L):
        if ode_order == 2:
            v_at_t = v[:, l].reshape((d, N))
            
            if xi is not None:
                xi_at_t = xi[:, l].reshape((-1, 1))
            else:
                xi_at_t = None
            
            external[:, l] = v_external(v_at_t, xi_at_t)
    
    external = external.flatten()
    
    return external
