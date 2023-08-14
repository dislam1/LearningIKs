import numpy as np
from scipy.interpolate import CubicSpline

def one_regularize_phihat_and_L2rhoT_errs(phi, phihat, phiknots, degree, sys_info, obs_info, learn_info, rhoLTemp):
    reg_output = regularize_phihat(phi, phihat, phiknots, degree, rhoLTemp, sys_info)
    phihatsmooth = reg_output['phihatsmooth']
    phihat_diff = reg_output['phihat_diff']
    phihatsmooth_diff = reg_output['phihatsmooth_diff']
    basis_info = reg_output['basis_info']
    L2rhoT_time = 0.0

    phi_L2norms = L2_rho_T_energy(phi, sys_info, obs_info, basis_info)
    phihat_L2rhoTdiff = L2_rho_T_energy(phihat_diff, sys_info, obs_info, basis_info)
    phihatsmooth_L2rhoTdiff = L2_rho_T_energy(phihatsmooth_diff, sys_info, obs_info, basis_info)

    Rel = phihat_L2rhoTdiff / phi_L2norms
    Rel_smooth = phihatsmooth_L2rhoTdiff / phi_L2norms
    Abs = phihat_L2rhoTdiff
    Abs_smooth = phihatsmooth_L2rhoTdiff

    output = {'L2rhoT_time': L2rhoT_time, 'Err': {'Rel': Rel, 'Rel_smooth': Rel_smooth, 'Abs': Abs, 'Abs_smooth': Abs_smooth}, 'phihatsmooth': phihatsmooth}
    return output

# Assuming the functions regularize_phihat and L2_rho_T_energy are already implemented
# and other required data is available as inputs to this function.

# Example usage:
# result = one_regularize_phihat_and_L2rhoT_errs(phi, phihat, phiknots, degree, sys_info, obs_info, learn_info, rhoLTemp)
