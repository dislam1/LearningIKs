import numpy as np

def find_xi_external(x, xi, sys_info):
    L = x.shape[1]
    external = np.zeros(xi.shape)
    
    for l in range(L):
        x_at_t = x[:, l].reshape((sys_info.d, sys_info.N))
        xi_at_t = xi[:, l].reshape((-1, 1))
        external[:, l] = sys_info.Fxi(xi_at_t, x_at_t)
    
    external = external.flatten()
    
    return external
