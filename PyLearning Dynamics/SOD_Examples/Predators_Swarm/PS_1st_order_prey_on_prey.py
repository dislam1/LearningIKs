import numpy as np

def PS_1st_order_prey_on_prey(r, prey_attract_prey):
    MAURO = False
    #The following code will replace zero with median value
    r = np.array(r)
    med = np.median(r[r > 0]) #Normalize the initial value if there is any zero.
    r[r == 0] = med

    
    if not MAURO:
        f = prey_attract_prey - np.power(r, -2)
    else:
        f = prey_attract_prey - np.power(r, -2)

    return f
