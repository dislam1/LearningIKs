import numpy as np

def general_riemann_sum(integrand, time_vec, kind):
    # Find out the number of elements in time_vec
    num_elms = len(time_vec)
    # Find out the number of sub-intervals
    num_subin = num_elms - 1
    # The integrand should have the same length as time_vec
    if len(integrand) != num_elms:
        raise ValueError('The length of the integrand must be the same as the length of time_vec.')

    # Find out the time step sizes
    time_steps = time_vec[1:] - time_vec[:-1]
    
    # Find out the Riemann sum based on kind
    if kind == 1:
        # The left Riemann sum
        r_sum = np.sum(integrand[:-1] * time_steps)
    elif kind == 2:
        # The right Riemann sum
        r_sum = np.sum(integrand[1:] * time_steps)
    elif kind == 3:
        # The Trapezoidal rule (kind of like a midpoint rule)
        r_sum = np.sum((integrand[1:] + integrand[:-1]) * time_steps) / 2
    else:
        raise ValueError('The function can only do three different approximations!!')
    
    return r_sum
