import numpy as np

def PS_1st_order_prey_on_predator(r, prey_attract_predator, predator_sense_prey):
    r = np.array(r)
    m = np.mean(r[r> 0])
# Replace the mean to the zero elements 
    r[r == 0] = m
    f = prey_attract_predator * r**(-predator_sense_prey)
    return f
