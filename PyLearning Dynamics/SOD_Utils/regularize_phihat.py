from SOD_Utils.simplifyfcn import simplifyfcn
from SOD_Utils.getFcnSupp import getFcnSupp
from SOD_Utils.intersectinterval import intersectinterval

def regularize_phihat(phi, phihat, phiknots, degree, rhoLTemp, sys_info):
    """
    Regularizes the estimator phihat and computes differences with the original estimator phi.

    Parameters:
        phi (dict): The original estimator dictionary.
        phihat (dict): The learned estimator dictionary.
        phiknots (dict): The knot vectors for the estimator basis functions.
        degree (dict): The degrees of the basis functions.
        rhoLTemp (dict): The dictionary containing temporary information about the density estimate.
        sys_info (dict): Additional information about the system.

    Returns:
        output (dict): The dictionary containing the regularized estimator, basis information, and differences.
    """
    import numpy as np

    basis_info = {}
    phihatsmooth = {}
    phihat_diff = {}
    phihatsmooth_diff = {}
    basis_info['phiknots']= phiknots

    for k_1 in range(sys_info['K'], -1, -1):
        for k_2 in range(sys_info['K'], -1, -1):
            basis_info[(k_1, k_2)]['supp'] = getFcnSupp(phihat[(k_1, k_2)], phiknots[(k_1, k_2)])
            basis_info[(k_1, k_2)]['interval'] = intersectinterval(basis_info[(k_1, k_2)]['supp'], rhoLTemp[(k_1, k_2)]['supp'])


            if basis_info[(k_1, k_2)]['interval'][1] - basis_info[(k_1, k_2)]['interval'][0] > 0:
                phihatsmooth[(k_1, k_2)] = simplifyfcn(phihat[(k_1, k_2)], phiknots[(k_1, k_2)], basis_info[(k_1, k_2)]['interval'],
                                                        degree[(k_1, k_2)])
            else:
                phihatsmooth[(k_1, k_2)] = phihat[(k_1, k_2)]

            phihat_diff[(k_1, k_2)] = lambda r: phi[(k_1, k_2)](r) - phihat[(k_1, k_2)](r)
            phihatsmooth_diff[(k_1, k_2)] = lambda r: phi[(k_1, k_2)](r) - phihatsmooth[(k_1, k_2)](r)

    output = {
        'basis_info': basis_info,
        'phihatsmooth': phihatsmooth,
        'phihat_diff': phihat_diff,
        'phihatsmooth_diff': phihatsmooth_diff
    }
    return output
