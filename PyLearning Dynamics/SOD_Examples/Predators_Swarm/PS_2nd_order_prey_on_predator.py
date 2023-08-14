import numpy as np

def PS_2nd_order_prey_on_predator(r):
    f = np.zeros_like(r)
    ind = r > 0
    # f[ind] = 1 / (r[ind] * (r[ind]**5 + 1))  # Original commented interaction
    # The commented interaction produces non-interesting dynamics
    f[ind] = 1.5 * r[ind]**(-2.5)
    ind = r == 0
    f[ind] = np.inf
    return f
