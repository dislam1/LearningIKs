import numpy as np

def ecdfinterp(Z, x):
    f, xe = np.histogram(Z, bins='auto', density=True)
    xcdf = np.interp(x, xe[1:], f, left=0, right=1)
    xcdf[x < min(Z)] = 0
    xcdf[x >= max(Z)] = 1
    return xcdf
