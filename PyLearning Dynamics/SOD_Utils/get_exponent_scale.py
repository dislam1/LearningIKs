import numpy as np

def get_exponent_scale(some_max):
    if some_max < 0:
        raise ValueError('Input: some_max has to be positive!!')
    if some_max > 1:
        Escale = int(np.ceil(np.log10(some_max)))
    elif some_max == 1:
        Escale = 1
    else:
        Escale = int(np.floor(np.log10(some_max)))
    return Escale
