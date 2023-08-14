import numpy as np
from SOD_Utils.standard_basis import standard_basis
def eval_basis_functions(x, alphas, vs):
    D = len(alphas)
    y = np.zeros_like(x)
    yprime = np.zeros_like(x)

    for i in range(D):
        if alphas[i] != 0:
            v, vprime = eval(vs['f'][i])(x)
            if np.count_nonzero(v) > 0:
                y += alphas[i] * v
            if np.count_nonzero(vprime) > 0:
                yprime += alphas[i] * vprime
                          
            

    return y, yprime
