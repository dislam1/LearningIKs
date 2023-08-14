import numpy as np
from scipy.sparse import spdiags

def find_pair_angle(x, v):
    """
    Calculates the pair angles between position differences and velocity vectors
    
    Args:
    x: x_i's, state of agents (position, opinions, etc.)
    v: \dot{x}_i's, derivative of the state

    Returns:
    theta: Pair angles between position differences and velocity vectors

    """
    d, N = x.shape
    v_vec = v.reshape((d * N, 1))
    x_pdiff = find_pair_diff(x)
    v_mat = np.tile(v_vec, (1, N))
    the_prod = x_pdiff * v_mat
    xdiff_dot_v = the_prod[np.arange(0, (N - 1) * d + 1, d), :]
    for d_idx in range(2, d + 1):
        xdiff_dot_v += the_prod[np.arange(d_idx - 1, (N - 1) * d + d_idx, d), :]
    
    pdist_mat = np.sqrt(np.abs(sqdist_mod(x)))
    pdist_zero = pdist_mat == 0
    v_norm = np.sqrt(np.sum(v**0.5, axis=0))
    vnorm_zero = v_norm == 0
    the_factors = spdiags(v_norm, 0, N, N) * pdist_mat
    the_factors[pdist_zero] = 1
    the_factors[vnorm_zero, :] = 1
    theta = np.arccos(xdiff_dot_v / the_factors)
    
    return theta
