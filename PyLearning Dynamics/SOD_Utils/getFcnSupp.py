import numpy as np
from SOD_Utils.eval_basis_functions import eval_basis_functions
def getFcnSupp(f, pts):
    
    alpha_vec = f['alpha_vec']
    basis = f['basis']    
    fun = f['fun']
    pts = np.linspace(min(pts), max(pts), 1000)

    fvals = (lambda r : eval_basis_functions(r, alpha_vec, basis) )(pts)

   
    idx_first = np.argmax(np.abs(fvals) > 10 * np.finfo(float).eps)
    if idx_first < len(pts):
        I_1 = pts[idx_first]
    else:
        I_1 = pts[0]

    idx_last = len(pts) - np.argmax(np.flip(np.abs(fvals) > 10 * np.finfo(float).eps)) - 1
    if idx_last >= 0:
        I_2 = pts[idx_last]
    else:
        I_2 = pts[-1]

    return [I_1, I_2]
