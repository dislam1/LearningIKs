import numpy as np

def PS_1st_order_predator_on_prey(r, predator_repulse_prey):
    MAURO = False
    r = np.array(r)
    m = np.mean(r[r > 0])
    # Replace the mean to the zero elements 
    r[r == 0] = m

    if not MAURO:
        f = -predator_repulse_prey * r**(-2)
    else:
        f = -predator_repulse_prey * r**(-6)

    return f
