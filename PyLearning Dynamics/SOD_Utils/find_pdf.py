import numpy as np

def find_pdf(hist_count):
    K = hist_count.shape[0]
    hist_prob = np.zeros_like(hist_count)

    for k1 in range(K):
        for k2 in range(K):
            total_count_Ck1_Ck2 = np.linalg.norm(hist_count[k1, k2, :], ord=1)
            hist_prob[k1, k2, :] = hist_count[k1, k2, :] / total_count_Ck1_Ck2

    return hist_prob
