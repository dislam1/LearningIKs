import numpy as np

def MS_interactions(r, type_index):
    # Define your energy-based interaction function here
    # Replace this with the actual implementation
    return 0

def MS_friction(v, type_index):
    # Define your non-collective change to v function here
    # Replace this with the actual implementation
    return 0

def MS_init_config(d, N, type_info, type_index):
    # Define your distribution of initial conditions here
    # Replace this with the actual implementation
    return np.zeros((N, d))

def ModelSelection4_def():
    N_preys = 9
    N_predators = 1

    sys_info = {
        'name': 'ModelSelection4',
        'd': 2,
        'N': N_preys + N_predators,
        'phiE': {
            (1, 1): lambda r: MS_interactions(r, 3),
            (1, 2): lambda r: MS_interactions(r, 7),
            (2, 1): lambda r: MS_interactions(r, 8),
            (2, 2): lambda r: MS_interactions(r, 6),
        },
        'phiA': None,
        'K': 2,
        'ode_order': 2,
        'type_info': np.concatenate([np.ones(N_preys, dtype=int), 2*np.ones(N_predators, dtype=int)]),
        'agent_mass': np.ones(N_preys + N_predators),
        'kappa': np.ones(2),
        'RE': None,
        'RA': None,
        'Fv': lambda v, xi: MS_friction(v, 1),
        'has_xi': False,
        'has_noise': False,
        'mu0': lambda: MS_init_config(sys_info['d'], sys_info['N'], sys_info['type_info'], 3),
        'T_f': 1,
    }

    solver_info = {
        'time_span': [0, sys_info['T_f']],
    }

    obs_info = {
        'L': 250,
        'M': 250,
        'L_test': 250,
        'T_L': sys_info['T_f'],
        'time_vec': np.linspace(0, sys_info['T_f'], 250),
        'use_derivative': False,
        'hist_num_bins': 1000,
    }

    basis_info = {
        'n': np.array([[298, 150], [150, 2]]),
        'type': 'standard',
        'degree': np.array([[1, 1], [1, 0]]),
    }

    learn_info = {
        'Ebasis_info': basis_info,
        'Abasis_info': None,
    }

    Example = {
        'sys_info': sys_info,
        'solver_info': solver_info,
        'obs_info': obs_info,
        'learn_info': learn_info,
    }

    return Example

# Example usage:
example_data = ModelSelection4_def()
print(example_data)
