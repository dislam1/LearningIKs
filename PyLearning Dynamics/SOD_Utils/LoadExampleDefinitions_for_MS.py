import numpy as np
from pytictoc import TicToc

def MS_interactions(r, interaction_type):
    # Define the energy based interactions
    # ...

def MS_friction(v, interaction_type):
    # Define the non-collective change to v (friction)
    # ...

def MS_init_config(d, N, type_info, example_num):
    # Define the distribution of initial conditions
    # ...

def LoadExampleDefinitions_for_MS():
    # Common parameters
    solver_info = {
        'solver': '15s',
        'rel_tol': 1.0e-5,
        'abs_tol': 1.0e-6,
    }

    plot_info = {
        'legend_font_size': 32,
        'colorbar_font_size': 20,
        'title_font_size': 20,
        'title_font_name': 'Helvetica',
        'axis_font_size': 32,
        'axis_font_name': 'Helvetica',
        'traj_line_width': 2.0,
        'phi_line_width': 1.5,
        'phihat_line_width': 1.5,
        'rhotscalingdownfactor': 1,
        'showplottitles': False,
        'display_phihat': False,
        'display_interpolant': True,
        'T_L_marker_size': 2.0,
    }

    learn_info = {
        'solver_type': 'pinv',
        'is_parallel': False,
        'is_adaptive': False,
        'keep_obs_data': True,
        'Riemann_sum': 2,
    }

    Examples = []

    def add_example(sys_info, obs_info, basis_info):
        example = {
            'sys_info': sys_info,
            'solver_info': solver_info,
            'obs_info': obs_info,
            'learn_info': learn_info,
            'plot_info': plot_info,
        }
        Examples.append(example)

    t = TicToc()

    # Model Selection 1: Energy Based Interactions (only)
    sys_info_1 = {
        'name': 'ModelSelection1',
        'd': 2,
        'N': 10,
        'phiE': [lambda r: MS_interactions(r, 1)],
        'phiA': [lambda r: np.zeros_like(r)],
        'K': 1,
        'ode_order': 2,
        'type_info': np.ones(10, dtype=int),
        'agent_mass': np.ones(10),
        'kappa': 1,
        'RE': [],
        'RA': [],
        'Fv': lambda v, xi: MS_friction(v, 1),
        'has_xi': False,
        'has_noise': False,
        'mu0': lambda: MS_init_config(2, 10, np.ones(10, dtype=int), 1),
        'T_f': 10,
    }

    obs_info_1 = {
        'L': 200,
        'M': 200,
        'T_L': 10,
        'time_vec': np.linspace(0, 10, 200),
        'use_derivative': False,
        'hist_num_bins': 2000,
    }

    basis_info_1 = {
        'n': max(300, int(obs_info_1['L'] * obs_info_1['M'] * sys_info_1['N'] * sys_info_1['d'] / 1000)),
        'type': 'standard',
        'degree': 1,
    }

    add_example(sys_info_1, obs_info_1, basis_info_1)

    # Model Selection 2: Alignment Based Interactions (only)
    sys_info_2 = {
        'name': 'ModelSelection2',
        'd': 2,
        'N': 10, 
        # Fill in the rest of the system information for model selection 2
    }

    obs_info_2 = {
        'L': 200,
        'M': 200,
        'T_L': 10,
        # Fill in the rest of the observation information for model selection 2
    }

    basis_info_2 = {
        # Fill in the basis information for model selection 2
    }

    add_example(sys_info_2, obs_info_2, basis_info_2)

    # Model Selection 3: First Order Heterogeneous Dynamics (missing order information)
    sys_info_3 = {
        'name': 'ModelSelection3',
        'd': 2,
        'N': 10,
        # Fill in the rest of the system information for model selection 3
    }

    obs_info_3 = {
        'L': 250,
        'M': 250,
        'T_L': 1,
        # Fill in the rest of the observation information for model selection 3
    }

    basis_info_3 = {
        # Fill in the basis information for model selection 3
    }

    add_example(sys_info_3, obs_info_3, basis_info_3)

    # Model Selection 4: Second Order Heterogeneous Dynamics (missing order information)
    sys_info_4 = {
        'name': 'ModelSelection4',
        'd': 2,
        'N': 10,
        # Fill in the rest of the system information for model selection 4
    }

    obs_info_4 = {
        'L': 250,
        'M': 250,
        'T_L': 1,
        # Fill in the rest of the observation information for model selection 4
    }

    basis_info_4 = {
        # Fill in the basis information for model selection 4
    }

    add_example(sys_info_4, obs_info_4, basis_info_4)

    return Examples

# Usage
Examples = LoadExampleDefinitions_for_MS()
