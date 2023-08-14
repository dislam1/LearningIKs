import numpy as np

def prepare_hist_items(K, hist_num_bins, M, max_rs, max_dotrs, max_xis):
    histedgesR = [[None] * K for _ in range(K)]
    histbinwidthR = np.zeros((K, K))
    histedgesDR = []
    histbinwidthDR = []
    histedgesXi = []
    histbinwidthXi = []
    
    if max_dotrs is not None:
        histedgesDR = [[None] * K for _ in range(K)]
        histbinwidthDR = np.zeros((K, K))
    
    if max_xis is not None:
        histedgesXi = [[None] * K for _ in range(K)]
        histbinwidthXi = np.zeros((K, K))
    
    for k1 in range(K):
        for k2 in range(K):
            histedgesR[k1][k2] = np.linspace(0, max_rs[k1, k2], hist_num_bins + 1)
            histbinwidthR[k1, k2] = max_rs[k1, k2] / hist_num_bins
            
            if max_dotrs is not None:
                histedgesDR[k1][k2] = np.linspace(0, max_dotrs[k1, k2], hist_num_bins + 1)
                histbinwidthDR[k1, k2] = max_dotrs[k1, k2] / hist_num_bins
            
            if max_xis is not None:
                histedgesXi[k1][k2] = np.linspace(0, max_xis[k1, k2], hist_num_bins + 1)
                histbinwidthXi[k1, k2] = max_xis[k1, k2] / hist_num_bins
    
    histcountR = [None] * M
    histcountA = None
    histcountDR = None
    jhistcountXi = None
    histcountXi = None
    
    if max_dotrs is not None:
        histcountA = [None] * M
        histcountDR = [None] * M
    
    if max_xis is not None:
        jhistcountXi = [None] * M
        histcountXi = [None] * M
    
    return (histedgesR, histbinwidthR, histedgesDR, histbinwidthDR, histedgesXi, histbinwidthXi, histcountR, histcountA, histcountDR, jhistcountXi, histcountXi)
