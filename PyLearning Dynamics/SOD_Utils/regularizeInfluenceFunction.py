
import numpy as np
from SOD_Utils.getFcnSupp import getFcnSupp
from SOD_Utils.intersectInterval import intersectInterval
from SOD_Utils.simplifyfcn import simplifyfcn


def regularizeInfluenceFunction(phi, basis, rhoLTemp, sys_info):
    """
    Regularizes the influence functions based on the density estimate.

    Parameters:
        phi (dict): The influence functions.
        basis (dict): The basis functions.
        rhoLTemp (dict): Temporary information about the density estimate.
        sys_info (dict): Information about the dynamical system.

    Returns:
        f_reg (dict): The regularized influence functions.
        basis (dict): Updated basis functions after regularization.
    """
    f_reg = {}
    basis = np.array(basis)
    #rhoLTemp = np.array(rhoLTemp)

    for k_1 in range(sys_info['K'] -1, -1, -1):
        for k_2 in range(sys_info['K'] -1, -1, -1):
            basis[k_1, k_2]['supp'] = getFcnSupp(phi[k_1][k_2], basis[k_1, k_2]['knots'])
            basis[k_1, k_2]['interval'] = intersectInterval(basis[k_1, k_2]['supp'], rhoLTemp['supp'][k_1][ k_2])
            if basis[k_1, k_2]['interval'][1] - basis[k_1, k_2]['interval'][0] > 0:
                f_reg[k_1, k_2] = simplifyfcn(phi[k_1][k_2], basis[k_1, k_2]['knots'], basis[k_1, k_2]['interval'], basis[k_1, k_2]['degree'])
            else:
                f_reg[k_1, k_2] = phi[k_1, k_2]

    return f_reg, basis
