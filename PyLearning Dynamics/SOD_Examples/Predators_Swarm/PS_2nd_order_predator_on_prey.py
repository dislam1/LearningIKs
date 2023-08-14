import numpy as np

def PS_2nd_order_predator_on_prey(r):
    f = np.zeros_like(r)
    ind = r > 0
    f[ind] = -r[ind]**(-2)
    ind = r == 0
    f[ind] = -np.inf
    return f
