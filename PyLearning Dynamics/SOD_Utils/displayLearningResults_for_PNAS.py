def display_learning_results_for_PNAS(learning_output, sys_info, chosen_dynamics, obs_info, learn_info, plot_info):
    # Compare the true interactions vs. learned (and/or regularized) phis
    if sys_info['name'] == 'PhototaxisDynamics':
        visualize_phis_for_PT(learning_output, sys_info, obs_info, plot_info)
    else:
        visualize_phis(learning_output, sys_info, obs_info, learn_info, plot_info)

    # Compare true trajectories vs. learned trajectories
    visualize_trajs_for_PNAS(learning_output, sys_info, chosen_dynamics, obs_info, plot_info)
