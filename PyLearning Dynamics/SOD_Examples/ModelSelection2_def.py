import numpy as np

def MS_interactions(r, type_index):
    # Define your alignment-based interaction function here
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

def ModelSelection2_def():
    sys_info = {
        'name': 'ModelSelection2',
        'd': 2,
        'N': 10,
        'phiE': [lambda r: np.zeros_like(r)],
        'phiA': [lambda r: MS_interactions(r, 2)],
        'K': 1,
        'ode_order': 2,
        'type_info': np.ones(10, dtype=int),
        'agent_mass': np.ones(10),
        'kappa': 1,
        'RE': None,
        'RA': None,
        'Fv': lambda v, xi: MS_friction(v, 1),
        'has_xi': False,
        'has_noise': False,
        'mu0': lambda: MS_init_config(sys_info['d'], sys_info['N'], sys_info['type_info'], 1),
        'T_f': 10,
    }

    solver_info = {
        'time_span': [0, sys_info['T_f']],
    }

    obs_info = {
        'L': 200,
        'M': 200,
        'T_L': sys_info['T_f'],
        'time_vec': np.linspace(0, sys_info['T_f'], 200),
        'use_derivative': False,
        'hist_num_bins': 2000,
    }

    basis_info = {
        'n': max(300, int(np.ceil(obs_info['L'] * obs_info['M'] * sys_info['N'] * sys_info['d'] / 1000))),
        'type': 'standard',
        'degree': 1,
    }

    learn_info = {
        'Ebasis_info': basis_info,
        'Abasis_info': basis_info,
    }

    Example = {
        'sys_info': sys_info,
        'solver_info': solver_info,
        'obs_info': obs_info,
        'learn_info': learn_info,
    }

    return Example

# Example usage:
example_data = ModelSelection2_def()
print(example_data)
