from SOD_Utils.regularizeInfluenceFunction import regularizeInfluenceFunction

def regularize_estimator(Estimator, rhoLTemp, learn_info):
    """
    Regularizes the estimator.

    Parameters:
        Estimator (dict): The estimator dictionary containing the learned estimator and basis functions.
        rhoLTemp (dict): The dictionary containing temporary information about the density estimate.
        learn_info (dict): Additional information about the learning process.

    Returns:
        new_Estimator (dict): The updated estimator dictionary with the regularized estimator and basis functions.
    """
    import numpy as np

    new_Estimator = Estimator.copy()

    # Regularize the three kinds of interaction laws and save the output
    if 'phiEhat' in Estimator and Estimator['phiEhat'] is not None:
        phiEhatsmooth, Ebasis2 = regularizeInfluenceFunction(Estimator['phiEhat'], Estimator['Ebasis'],
                                                              rhoLTemp['rhoLTE'], learn_info['sys_info'])
        new_Estimator['phiEhatsmooth'] = phiEhatsmooth
        new_Estimator['Ebasis2'] = Ebasis2
    else:
        new_Estimator['phiEhatsmooth'] = None
        new_Estimator['Ebasis2'] = None

    if 'phiAhat' in Estimator and len(Estimator['phiAhat'][0]) > 0:
        phiAhatsmooth, Abasis2 = regularizeInfluenceFunction(Estimator['phiAhat'], Estimator['Abasis'],
                                                              rhoLTemp['rhoLTA'], learn_info['sys_info'])
        new_Estimator['phiAhatsmooth'] = phiAhatsmooth
        new_Estimator['Abasis2'] = Abasis2
    else:
        new_Estimator['phiAhatsmooth'] = None
        new_Estimator['Abasis2'] = None

    if 'phiXihat' in Estimator and Estimator['phiXihat'] is not None:
        phiXihatsmooth, Xibasis2 = regularizeInfluenceFunction(Estimator['phiXihat'], Estimator['Xibasis'],
                                                                rhoLTemp['rhoLTXi'], learn_info['sys_info'])
        new_Estimator['phiXihatsmooth'] = phiXihatsmooth
        new_Estimator['Xibasis2'] = Xibasis2
    else:
        new_Estimator['phiXihatsmooth'] = None
        new_Estimator['Xibasis2'] = None

    return new_Estimator
