import numpy as np
from scipy.optimize import minimize

def solve_for_mass_and_phim(sys_info, learningOutput, method, solver=1):
    # Prepare the optimization option
    if solver == 1:
        options = {'algorithm': 'interior-point', 'jac': True, 'maxiter': 50000, 'maxfun': 50000, 'tol': 1e-8}
    elif solver == 2:
        options = {'algorithm': 'trust-ncg', 'jac': True, 'maxiter': 50000, 'maxfun': 50000, 'tol': 1e-8}
    else:
        raise ValueError("Invalid solver option. Choose 1 or 2.")

    # Optimization Approach to discovering the mass
    gravity_terms = generateGravityMat(learningOutput, method)
    rp = gravity_terms['rp']
    P = len(rp)

    # Initialize storage
    y = np.zeros(sys_info['N'] + P)

    if method == 1:
        # Find (\beta_1, \vec{\phi}_m) first
        params = gravity_terms['Phii1Mat'], gravity_terms['Rhoi1Mat']
        obj_func = lambda x: total_gravity_energy(x, params, 1)
        x0 = np.ones(P + 1)
        x = minimize(obj_func, x0, constraints={'type': 'ineq', 'fun': lambda x: -x[0]}, bounds=[(0, None)] * (P + 1),
                     method='SLSQP', options=options)['x']
        # Unpack the data in x, and put it back in y, the first N entries of Y are for beta,
        # and the remaining P entries are for phi_vec
        y[0] = x[0]
        phi_vec = x[1:]
        y[sys_info['N']:] = phi_vec

        # Find (\beta_2, \cdots, \beta_N) next
        params = gravity_terms['Phi1iMat'], gravity_terms['Rho1iMat'], phi_vec
        obj_func = lambda x: total_gravity_energy(x, params, 2)
        x0 = np.ones(sys_info['N'] - 1)
        x = minimize(obj_func, x0, constraints={'type': 'ineq', 'fun': lambda x: -x}, bounds=[(0, None)] * (sys_info['N'] - 1),
                     method='SLSQP', options=options)['x']
        y[1:sys_info['N']] = x
    elif method == 2:
        # Find \beta's and phi_vec together
        params = gravity_terms['Phii1Mat'], gravity_terms['Rhoi1Mat'], gravity_terms['Phi1iMat'], gravity_terms['Rho1iMat']
        obj_func = lambda x: total_gravity_energy(x, params, 3)
        x0 = np.ones(sys_info['N'] + P) / (sys_info['N'] + P)
        x = minimize(obj_func, x0, constraints={'type': 'ineq', 'fun': lambda x: -x}, bounds=[(0, None)] * (sys_info['N'] + P),
                     method='SLSQP', options=options)['x']
        y = x
    elif method == 3:
        # Find (\beta_1, \cdots, \beta_N) and phi_vec all together from all phi's and rho's
        params = gravity_terms['PhiMat'], gravity_terms['RhoMat']
        obj_func = lambda x: total_gravity_energy(x, params, 4)
        x0 = np.ones(sys_info['N'] + P) / (sys_info['N'] + P)
        x = minimize(obj_func, x0, constraints={'type': 'ineq', 'fun': lambda x: -x}, bounds=[(0, None)] * (sys_info['N'] + P),
                     method='SLSQP', options=options)['x']
        y = x
    else:
        raise ValueError("Invalid method option. Choose 1, 2, or 3.")

    return y, rp
