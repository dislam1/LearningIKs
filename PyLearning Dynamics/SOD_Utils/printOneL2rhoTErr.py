import numpy as np
def printOneL2rhoTErr(K, Err, ErrSmooth, kind):
    for k1 in range(K):
        for k2 in range(K):
            rel_err_original = [x.Rel[k1, k2] for x in Err]
            rel_err_smooth = [x.Rel[k1, k2] for x in ErrSmooth]
            abs_err_original = [x.Abs[k1, k2] for x in Err]
            abs_err_smooth = [x.Abs[k1, k2] for x in ErrSmooth]

            print(f"\n------------------- {kind} Based Interaction L2(rho_T) Errors -- Relative Errors:")
            print(f"\tRelative L_2(rho_T) error of original learned estimator for phi_{{{k1}, {k2}}} = "
                  f"{np.mean(rel_err_original):10.4e}±{np.std(rel_err_original):10.4e}.")
            print(f"\tRelative L_2(rho_T) error of smooth learned estimator for phi_{{{k1}, {k2}}} = "
                  f"{np.mean(rel_err_smooth):10.4e}±{np.std(rel_err_smooth):10.4e}.")

            print(f"\n------------------- {kind} Based Interaction L2(rho_T) Errors -- Absolute Errors:")
            print(f"\tAbsolute L_2(rho_T) error of original learned estimator for phi_{{{k1}, {k2}}} = "
                  f"{np.mean(abs_err_original):10.4e}±{np.std(abs_err_original):10.4e}.")
            print(f"\tAbsolute L_2(rho_T) error of smooth learned estimator for phi_{{{k1}, {k2}}} = "
                  f"{np.mean(abs_err_smooth):10.4e}±{np.std(abs_err_smooth):10.4e}.")


# Example usage:
# Assuming you have the variables K, Err, ErrSmooth, and kind
# printOneL2rhoTErr(K, Err, ErrSmooth, kind)
