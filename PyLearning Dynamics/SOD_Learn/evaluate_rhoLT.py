import numpy as np
from scipy.interpolate import interp1d

def evaluate_rhoLT(hist, histedges, supp, r):
    # function y = evaluate_rhoLT(hist, histedges, r)
    #
    # (c) M. Zhong (JHU)

    edges = histedges[:-1]
    binwidth = (edges[1] - edges[0]) / len(edges)
    edges_idxs = np.where(hist > 0.01 * binwidth)[0]
    edges_idxs = np.arange(edges_idxs.min(), edges_idxs.max() + 1)
    histdata, edges = downsampleHistCounts(hist[edges_idxs[:-1]], edges[edges_idxs], int(np.sqrt(len(edges_idxs))) / 2)
    centers = (edges[:-1] + edges[1:]) / 2
    densRhoLT = interp1d(centers, histdata, kind='linear', fill_value=0, bounds_error=False)
    ind = (supp[0] <= r) & (r <= supp[1])
    y = np.zeros_like(r)
    y[ind] = densRhoLT(r[ind])
    return y

'''
Please note that the code assumes that the function downsampleHistCounts() 
is defined elsewhere in your code. 
You'll need to provide its definition for the code to work properly.
'''