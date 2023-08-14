import numpy as np

def get_range_from_rhos(rhoLTemp):
    total_num_trials = len(rhoLTemp)
    max_rs = np.zeros(total_num_trials)
    min_rs = np.zeros(total_num_trials)
    for ind in range(total_num_trials):
        min_rs[ind] = rhoLTemp[ind]['supp'][0]
        max_rs[ind] = rhoLTemp[ind]['supp'][1]
    max_r = np.max(max_rs)
    min_r = np.min(min_rs)
    if max_r < min_r + 10 * np.finfo(float).eps:
        max_r = min_r + 1
        min_r = min_r - 1
    range_val = [min_r, max_r]
    return range_val
