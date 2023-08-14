import numpy as np
from pytictoc import TicToc
import os
import importlib

def LoadExampleDefinitions(kind='Main'):
    # Common parameters
    # for solver_info (ODE integrator)
    #solver = '15s'
    solver = 'SCIPY ODE Integrator'
    rel_tol = 1.0e-5
    abs_tol = 1.0e-6

    # visualization
    plot_info = {
        'scrsz': [1, 1, 1920, 1080],
        'legend_font_size': 14,
        'legend_font_name': 'arial',
        'colorbar_font_size': 20,
        'title_font_size': 26,
        'title_font_name': 'arial',
        'axis_font_size': 14,
        'axis_font_name': 'arial',
        'tick_font_size': 14,
        'tick_font_name': 'arial',
        'traj_line_width': 2.0,
        'phi_line_width': 1.5,
        'phihat_line_width': 1.5,
        'rhotscalingdownfactor': 1,
        'showplottitles': False,
        'display_phihat': False,
        'display_interpolant': False,
        'for_PNAS': False,
        'line_styles': ['-', '-.', '--', ':'],
        'T_L_marker_size': 2.0,
    }
    #'display_interpolant': True, turn off for time being

    # for learn_info
    solver_type = 'pinv'
    is_parallel = False
    is_adaptive = False
    keep_obs_data = True
    Riemann_sum = 2
    N_ratio = 4

    # find all the files
    #Define the directory for the programs
    prog_parent_dir = os.path.expanduser('~/Downloads/Maggioni Python')
    def_files = [ f for f in os.listdir(prog_parent_dir+'/SOD_Examples') if f.endswith('_def.py')]

    switcher = {
        'Main': lambda name: 'ModelSelection' not in name,
        'ModelSelection': lambda name: 'ModelSelection' in name
    }
    
    def_files = [def_file for def_file in def_files if switcher.get(kind, lambda name: False)(def_file)]
    
    total_num_defs = len(def_files)
    Examples = []

    t = TicToc()
    for idx in range(total_num_defs):
        t.tic()
        module_name = os.path.splitext(os.path.basename(def_files[idx]))[0]
        Example = importlib.import_module("SOD_Examples."+module_name).__dict__[module_name]()
        Example['plot_info'] = plot_info.copy()
        #Example['solver_info'] = {'solver': solver, 'rel_tol': rel_tol, 'abs_tol': abs_tol}
        Example['solver_info']['solver'] = solver
        Example['solver_info']['rel_tol'] = rel_tol
        Example['solver_info']['abs_tol'] = abs_tol

        #Example['learn_info'] = {'solver_type': solver_type, 'is_parallel': is_parallel, 'is_adaptive': is_adaptive,
         #                     'keep_obs_data': keep_obs_data, 'Riemann_sum': Riemann_sum, 'N_ratio': N_ratio}
        Example['learn_info']['solver_type']=solver_type
        Example['learn_info']['is_parallel']=is_parallel
        Example['learn_info']['is_adaptive']=is_adaptive
        Example['learn_info']['keep_obs_data']=keep_obs_data
        Example['learn_info']['Riemann_sum']=Riemann_sum
        Example['learn_info']['N_ratio']=N_ratio

        Examples.append(Example)
        t.toc('Loaded example {}/{} in '.format(idx + 1, total_num_defs))
    return Examples
