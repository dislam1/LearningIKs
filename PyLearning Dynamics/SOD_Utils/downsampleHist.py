import numpy as np

def downsample_hist(hist, hist_edges, factor):
    n_bins = len(hist_edges)
    idxs = np.unique(np.floor(np.linspace(1, n_bins, n_bins // factor)))
    d_hist_edges = hist_edges[idxs]
    d_hist_count = np.zeros(len(idxs) - 1)
    bin_wid_old = hist_edges[1] - hist_edges[0]
    bin_wid_new = d_hist_edges[1] - d_hist_edges[0]
    for k in range(len(idxs) - 1):
        d_hist_count[k] = np.sum(hist[idxs[k]:idxs[k + 1] - 1])
    d_hist_count = d_hist_count * bin_wid_old / bin_wid_new
    return d_hist_count, d_hist_edges
