import numpy as np
from scipy.sparse import spdiags
from SOD_Learn.get_class_modifier import get_class_modifier
from SOD_Learn.get_Riemann_based_time_modifier import get_Riemann_based_time_modifier

def scale_the_quantities(the_F, d_vec, energy_Phi, align_Phi, N, num_classes, class_info, time_vec, Riemann):
    L = len(time_vec)
    d = d_vec.shape[0] // (L * N)
    class_mod = get_class_modifier(L, N, d, num_classes, class_info)
    
    if L == 1:
        time_mod = np.ones(N * d)
        D_mod = spdiags(time_mod / class_mod, 0, N * d, N * d)
    else:
        time_mod = get_Riemann_based_time_modifier(N, d, time_vec, Riemann)
        D_mod = spdiags(time_mod / class_mod, 0, L * N * d, L * N * d)
    
    the_F = D_mod @ the_F
    d_vec = D_mod @ d_vec
    
    if len(energy_Phi) > 0:
        energy_Phi = D_mod @ energy_Phi
    
    if len(align_Phi) > 0:
        align_Phi = D_mod @ align_Phi
    
    return the_F, d_vec, energy_Phi, align_Phi
