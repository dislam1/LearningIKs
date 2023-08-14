import numpy as np

def PS_2nd_order_prey_on_prey(r):
    f = 1 - r**(-2)
    return f

def PS_2nd_order_predator_on_prey(r):
    f = 1.5 * r**(-2.5)
    return f

def PS_2nd_order_prey_on_predator(r):
    f = np.zeros_like(r)
    ind = r > 0
    f[ind] = 1./(r[ind] * (r[ind]**(5) + 1))
    ind = r == 0
    f[ind] = np.Inf
    return f

def PS_2nd_order_predator_on_predator(r):
    f = np.zeros_like(r)
    return f

def PS_2nd_order_friction(v, nus, num_classes, class_info):
    d, N = v.shape
    friction = np.zeros((d, N))
    for k in range(1, num_classes + 1):
        agents_Ck1 = class_info == k
        friction[:, agents_Ck1] = -nus[k-1] * v[:, agents_Ck1]
    friction = friction.ravel()
    return friction

def PS_init_config(N, type_info, kind):
    # Define the uniform_dist function (not shown here) to generate initial positions
    # This function should be defined as in the previous section.
    pass

def PredatorPrey2ndOrder_def():
    # System
    sys_info = {}
    N_preys = 9
    N_predators = 1
    N = N_preys + N_predators
    sys_info['name'] = 'PredatorPrey2ndOrder'
    sys_info['d'] = 2
    sys_info['N'] = N_preys + N_predators
    sys_info['phiE'] = {
        (1, 1): lambda r: PS_2nd_order_prey_on_prey(r),
        (1, 2): lambda r: PS_2nd_order_predator_on_prey(r),
        (2, 1): lambda r: PS_2nd_order_prey_on_predator(r),
        (2, 2): lambda r: PS_2nd_order_predator_on_predator(r)
    }
    sys_info['phiA'] = None
    sys_info['K'] = 2
    sys_info['ode_order'] = 2
    sys_info['agent_mass'] = np.ones(N)
    sys_info['type_info'] = np.concatenate((np.ones(N_preys), 2 * np.ones(N_predators)))
    sys_info['kappa'] = np.ones(sys_info['K'])
    sys_info['RE'] = None
    sys_info['RA'] = None
    sys_info['Fv'] = lambda v, xi: PS_2nd_order_friction(v, np.array([1, 1]), 2, sys_info['type_info'])
    sys_info['has_xi'] = False
    sys_info['has_noise'] = False
    sys_info['mu0'] = lambda: PS_init_config(sys_info['N'], sys_info['type_info'], 2)
    sys_info['T_f'] = 20

    # ODE solver
    solver_info = {'time_span': [0, sys_info['T_f']]}

    # Observations
    obs_info = {}
    obs_info['L'] = 300
    obs_info['M'] = 150
    obs_info['M_rhoT'] = 2000
    obs_info['T_L'] = sys_info['T_f'] / 2
    obs_info['time_vec'] = np.linspace(0, obs_info['T_L'], obs_info['L'])
    obs_info['use_derivative'] = False
    obs_info['hist_num_bins'] = 10000
    obs_info['obs_noise'] = 0.1

    # Learning
    basis_info = {}
    basis_info['n'] = np.maximum(64 * np.ones((sys_info['K'], sys_info['K'])),
                                np.array([[np.ceil(obs_info['L'] * obs_info['M'] * N_preys * sys_info['d'] / 500),
                                           np.ceil(obs_info['L'] * obs_info['M'] * np.sqrt(N_preys * N_predators) *
                                                   sys_info['d'] / 500)],
                                          [np.ceil(obs_info['L'] * obs_info['M'] * np.sqrt(N_preys * N_predators) *
                                                   sys_info['d'] / 500),
                                           np.ceil(obs_info['L'] * obs_info['M'] * N_predators * sys_info['d'] / 500)]]))
    basis_info['type'] = 'standard'
    basis_info['degree'] = np.array([[1, 1], [1, 0]])
    learn_info = {'Ebasis_info': basis_info, 'Abasis_info': None}

    # package the data
    Example = {}
    Example['sys_info'] = sys_info
    Example['solver_info'] = solver_info
    Example['obs_info'] = obs_info
    Example['learn_info'] = learn_info

    return Example
