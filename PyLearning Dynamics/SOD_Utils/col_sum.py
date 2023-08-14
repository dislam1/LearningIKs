import numpy as np

def col_sum(x):
    # Sum for each column.
    # A more readable alternative to np.sum(x, axis=0).
    s = np.sum(x, axis=0)
    return s
