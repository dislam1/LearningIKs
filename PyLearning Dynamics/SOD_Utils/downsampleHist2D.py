import numpy as np

def downsample_hist_2d(hist, hist_edges_1, factor_1, hist_edges_2, factor_2):
    n_bins_1 = len(hist_edges_1)
    idxs_1 = np.unique(np.floor(np.linspace(1, n_bins_1, n_bins_1 // factor_1)))
    d_hist_edges_1 = hist_edges_1[idxs_1]
    
    n_bins_2 = len(hist_edges_2)
    idxs_2 = np.unique(np.floor(np.linspace(1, n_bins_2, n_bins_2 // factor_2)))
    d_hist_edges_2 = hist_edges_2[idxs_2]
    
    d_hist_count = np.zeros((len(idxs_1) - 1, len(idxs_2) - 1))
    bin_wid_1_old = hist_edges_1[1] - hist_edges_1[0]
    bin_wid_2_old = hist_edges_2[1] - hist_edges_2[0]
    bin_wid_1_new = d_hist_edges_1[1] - d_hist_edges_1[0]
    bin_wid_2_new = d_hist_edges_2[1] - d_hist_edges_2[0]
    
    for k1 in range(len(idxs_1) - 1):
        for k2 in range(len(idxs_2) - 1):
            d_hist_count[k1, k2] = np.sum(np.sum(hist[idxs_1[k1]:idxs_1[k1 + 1] - 1, idxs_2[k2]:idxs_2[k2 + 1] - 1]))
    
    d_hist_count = d_hist_count * bin_wid_1_old * bin_wid_2_old / (bin_wid_1_new * bin_wid_2_new)
    return d_hist_count, d_hist_edges_1, d_hist_edges_2
