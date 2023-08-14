import numpy as np

def PS_1st_order_prey_on_prey(r, prey_attract_prey):
    f = prey_attract_prey - r**(-2)
    return f

def PS_1st_order_predator_on_prey(r, predator_repulse_prey):
    f = -predator_repulse_prey * r**(-2)
    return f

def PS_1st_order_prey_on_predator(r, prey_attract_predator, predator_sense_prey):
    f = prey_attract_predator - r**(-predator_sense_prey)
    return f

def PS_1st_order_predator_on_predator(r):
    f = np.zeros_like(r)
    return f

def PS_init_config(N, type_info, kind):
    # Define the uniform_dist function (not shown here) to generate initial positions
    # This function should be defined as in the previous section.
    pass

def PredatorPrey1stOrderSplines_def():
    # System
    sys_info = {}
    sys_info['name'] = 'PredatorPrey1stOrderSplines'
    N_preys = 9
    N_predators = 1
    prey_attract_prey = 1
    predator_repulse_prey = 2
    prey_attract_predator = 3.5
    predator_sense_prey = 3
    sys_info['d'] = 2
    sys_info['N'] = N_preys + N_predators
    sys_info['phiE'] = {
        (1, 1): lambda r: PS_1st_order_prey_on_prey(r, prey_attract_prey),
        (1, 2): lambda r: PS_1st_order_predator_on_prey(r, predator_repulse_prey),
        (2, 1): lambda r: PS_1st_order_prey_on_predator(r, prey_attract_predator, predator_sense_prey),
        (2, 2): lambda r: PS_1st_order_predator_on_predator(r)
    }
    sys_info['K'] = 2
    sys_info['ode_order'] = 1
    sys_info['type_info'] = np.concatenate((np.ones(N_preys), 2 * np.ones(N_predators)))
    sys_info['kappa'] = np.ones(sys_info['K'])
    sys_info['RE'] = None
    sys_info['has_noise'] = False
    sys_info['mu0'] = lambda: PS_init_config(sys_info['N'], sys_info['type_info'], 1)
    sys_info['T_f'] = 10

    # ODE solver
    solver_info = {'time_span': [0, sys_info['T_f']]}

    # Observations
    obs_info = {}
    obs_info['L'] = 200
    obs_info['M'] = 50
    obs_info['M_rhoT'] = 2000
    obs_info['T_L'] = sys_info['T_f'] / 2
    obs_info['time_vec'] = np.linspace(0, obs_info['T_L'], obs_info['L'])
    obs_info['use_derivative'] = True
    obs_info['hist_num_bins'] = 10000
    obs_info['obs_noise'] = 0.0

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
    basis_info['is_splines'] = True
    learn_info = {'Ebasis_info': basis_info}

    # package the data
    Example = {}
    Example['sys_info'] = sys_info
    Example['solver_info'] = solver_info
    Example['obs_info'] = obs_info
    Example['learn_info'] = learn_info

    return Example
