import numpy as np

def getHistSupp(bindistrib, histedges, quantthres=0.0):
    cumdistrib = np.cumsum(bindistrib)

    idx_first = np.argmax(cumdistrib > quantthres)
    if idx_first < len(histedges):
        estsupp_1 = histedges[idx_first]
    else:
        estsupp_1 = 0

    idx_last = len(histedges) - np.argmax(np.flip(cumdistrib) < (1 - quantthres) * cumdistrib[-1]) - 1
    if idx_last >= 0:
        estsupp_2 = histedges[idx_last]
    else:
        estsupp_2 = 0

    return [estsupp_1, estsupp_2]
