import numpy as np

def get_hist_supp(min_pt, max_pt, histedges):
    K = min_pt.shape[0]
    hist_supp = np.zeros((K, K, 2))

    for k1 in range(K):
        for k2 in range(K):
            hist_supp[k1, k2, 1] = histedges[k1, k2][np.searchsorted(histedges[k1, k2], min_pt[k1, k2], side='right') - 1]
            hist_supp[k1, k2, 2] = max_pt[k1, k2]

    return hist_supp
