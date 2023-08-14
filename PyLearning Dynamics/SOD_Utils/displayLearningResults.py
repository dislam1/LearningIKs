
from SOD_Utils.visualize_trajs import visualizetrajs
from SOD_Utils.visualize_phis import visualizephis
from SOD_Utils.visualize_phis_for_PT import visualizephisforPT

def displaylearningresults(learning_output, sys_info, sys_info_approx, obs_info, learn_info, ICs, plot_info):
    # Compare the true interactions vs. learned (and/or regularized) phis
    if sys_info['name'] == 'PhototaxisDynamics':
        visualizephisforPT(learning_output, sys_info, obs_info, plot_info)
    else:
        visualizephis(learning_output, sys_info, obs_info, learn_info, plot_info)

    # Compare true trajectories vs. learned trajectories
    visualizetrajs(learning_output, sys_info, sys_info_approx, obs_info, ICs, plot_info)
