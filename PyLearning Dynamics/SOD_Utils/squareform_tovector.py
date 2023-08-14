import numpy as np

def squareform_tovector(Y):
    n = Y.shape[0]
    Z = Y[np.tril_indices(n, k=-1)]
    return Z
