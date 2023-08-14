import numpy as np

def squareform_tomatrix(Y):
    n = Y.size
    m = int(np.ceil(np.sqrt(2 * n)))  # (1 + np.sqrt(1 + 8 * n)) / 2, but works for large n

    Z = np.zeros((m, m), dtype=Y.dtype)
    if m > 1:
        triu_indices = np.triu_indices(m, k=1)
        Z[triu_indices] = Y
        Z.T[triu_indices] = Y

    return Z
