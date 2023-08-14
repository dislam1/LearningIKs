import numpy as np
def check_rhoLT_independence(sys_info, rhoLT):
    # Check if the system ODE is second order
    if sys_info['ode_order'] != 2:
        raise Exception('System ODE has to be second order!!')
    else:
        if 'phiA' not in sys_info or not sys_info['phiA']:
            raise Exception('System has to have alignment-based interactions!!')

    the_diff = {'rhoLTA_diff': np.zeros((sys_info['K'], sys_info['K']))}

    for k1 in range(sys_info['K']):
        for k2 in range(sys_info['K']):
            bin_width1 = rhoLT['rhoLTA']['rhoLTR']['histedges'][k1, k2][1] - rhoLT['rhoLTA']['rhoLTR']['histedges'][k1, k2][0]
            bin_width2 = rhoLT['rhoLTA']['rhoLTDR']['histedges'][k1, k2][1] - rhoLT['rhoLTA']['rhoLTR']['histedges'][k1, k2][0]
            the_diff['rhoLTA_diff'][k1, k2] = calculate_independence(rhoLT['rhoLTA']['hist'][k1, k2], 
                                                                    rhoLT['rhoLTA']['rhoLTR']['hist'][k1, k2],
                                                                    rhoLT['rhoLTA']['rhoLTDR']['hist'][k1, k2],
                                                                    bin_width1, bin_width2)

    if sys_info['has_xi']:
        the_diff['rhoLTXi_diff'] = np.zeros((sys_info['K'], sys_info['K']))
        for k1 in range(sys_info['K']):
            for k2 in range(sys_info['K']):
                bin_width1 = rhoLT['rhoLTXi']['rhoLTR']['histedges'][k1, k2][1] - rhoLT['rhoLTA']['rhoLTR']['histedges'][k1, k2][0]
                bin_width2 = rhoLT['rhoLTXi']['mrhoLTXi']['histedges'][k1, k2][1] - rhoLT['rhoLTA']['rhoLTR']['histedges'][k1, k2][0]
                the_diff['rhoLTXi_diff'][k1, k2] = calculate_independence(rhoLT['rhoLTXi']['hist'][k1, k2], 
                                                                          rhoLT['rhoLTXi']['rhoLTR']['hist'][k1, k2],
                                                                          rhoLT['rhoLTXi']['mrhoLTXi']['hist'][k1, k2],
                                                                          bin_width1, bin_width2)

    return the_diff
