def regularize_and_compute_relative_error(learn_out, learn_info, obs_info, kind):
    """
    Regularizes the learned estimator and computes the relative error.

    Parameters:
        learn_out (dict): The learning output containing the learned estimator and other information.
        learn_info (dict): Additional information about the learning process.
        obs_info (dict): Observation information.
        kind (str): The type of interaction ('energy', 'alignment', or 'xi').

    Returns:
        new_learn_out (dict): The updated learning output with the regularized estimator and relative error.
    """
    import numpy as np

    # Pick a set of data
    if kind == 'energy':
        phi = learn_info['sys_info']['phiE']
        phihat = learn_out['Estimator']['phiEhat']
        basis = learn_out['Estimator']['Ebasis']
        rhoLTemp = learn_out['rhoLTemp']['rhoLTE']
    elif kind == 'alignment':
        phi = learn_info['sys_info']['phiA']
        phihat = learn_out['Estimator']['phiAhat']
        basis = learn_out['Estimator']['Abasis']
        rhoLTemp = learn_out['rhoLTemp']['rhoLTA']
    elif kind == 'xi':
        phi = learn_info['sys_info']['phiXi']
        phihat = learn_out['Estimator']['phiXihat']
        basis = learn_out['Estimator']['Xibasis']
        rhoLTemp = learn_out['rhoLTemp']['rhoLTXi']
    else:
        raise ValueError("Invalid kind, must be 'energy', 'alignment', or 'xi'")

    # Regularize, compute, and output data
    phihatsmooth, basis2 = regularize_influence_function(phihat, basis, rhoLTemp, learn_info['sys_info'])
    Err = relative_error_influence_function(phihat, phi, learn_info['sys_info'], obs_info, basis, kind)
    ErrSmooth = relative_error_influence_function(phihatsmooth, phi, learn_info['sys_info'], obs_info, basis, kind)

    if learn_info['VERBOSE'] >= 1:
        print(f"\n------------------- For {kind} based interactions")
        for k1 in range(Err['Rel'].shape[0]):
            for k2 in range(Err['Rel'].shape[1]):
                if Err['Rel'][k1, k2] < np.inf:
                    print(f"Relative L_2(rho_T) error of original learned estimator for phi_{{{k1},{k2}}} = {Err['Rel'][k1, k2]:.6e}")
                    print(f"Relative L_2(rho_T) error of smooth   learned estimator for phi_{{{k1},{k2}}} = {ErrSmooth['Rel'][k1, k2]:.6e}")
                else:
                    print(f"Absolute L_2(rho_T) error of original learned estimator for phi_{{{k1},{k2}}} = {Err['Abs'][k1, k2]:.6e}")
                    print(f"Absolute L_2(rho_T) error of smooth   learned estimator for phi_{{{k1},{k2}}} = {ErrSmooth['Abs'][k1, k2]:.6e}")

    # Package data
    new_learn_out = learn_out.copy()
    if kind == 'energy':
        new_learn_out['Estimator']['phiEhatsmooth'] = phihatsmooth
        new_learn_out['Estimator']['Ebasis2'] = basis2
        new_learn_out['EErr'] = Err
        new_learn_out['EErrSmooth'] = ErrSmooth
    elif kind == 'alignment':
        new_learn_out['Estimator']['phiAhatsmooth'] = phihatsmooth
        new_learn_out['Estimator']['Abasis2'] = basis2
        new_learn_out['AErr'] = Err
        new_learn_out['AErrSmooth'] = ErrSmooth
    elif kind == 'xi':
        new_learn_out['Estimator']['phiXihatsmooth'] = phihatsmooth
        new_learn_out['Estimator']['Xibasis2'] = basis2
        new_learn_out['XiErr'] = Err
        new_learn_out['XiErrSmooth'] = ErrSmooth

    return new_learn_out
