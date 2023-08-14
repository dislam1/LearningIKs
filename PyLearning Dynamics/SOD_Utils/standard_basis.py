import numpy as np

def standard_basis(n, x, xspan):
    x = np.array(x)
    x_k, x_kp1 = xspan
    if x_kp1 <= x_k:
        print("Cannot continue...span values is not correct \n")
        return
    
    ind = np.where(np.logical_and(x >= x_k , x < x_kp1))[0]
    
    psi = np.zeros_like(x)
    dpsi = np.zeros_like(x)
    psi[ind] = np.power(x[ind], n)
    if n > 0:
        dpsi[ind] = n * np.power(x[ind], n - 1)
    return psi, dpsi
