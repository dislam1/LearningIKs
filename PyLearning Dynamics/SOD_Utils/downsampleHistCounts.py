import numpy as np

def downsampleHistCounts(hist, hist_edges, factor):
    n_bins = len(hist_edges)
    idxs = np.unique(np.floor(np.linspace(0, n_bins -1, (n_bins -1) // int(factor) )))
    idxs = [j.astype(int) for i, j in enumerate(idxs)]
    d_histedges = (hist_edges.ravel())[idxs]
    d_histcount = np.zeros(len(idxs) -1 )
    bin_wid_old = hist_edges[1] - hist_edges[0]
    bin_wid_new = d_histedges[1] - d_histedges[0]
    
    for k in range(len(idxs) -1):
        d_histcount[k] = np.sum(hist[idxs[k]:idxs[k + 1]-1 ])
    
    d_histcount = d_histcount * bin_wid_old / bin_wid_new
    return d_histcount, d_histedges
