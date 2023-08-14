import numpy as np

def colnorms(X):
    # Compute column norms of the input matrix X
    norms = np.sqrt(np.sum(X**2, axis=0))
    return norms
