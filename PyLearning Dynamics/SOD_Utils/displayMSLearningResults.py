import matplotlib.pyplot as plt

def display_MS_learning_results(learning_output, sys_info, plot_info):
    # Prepare the window size
    fig = plt.figure('Learned PhiE vs. PhiA', figsize=(12, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    total_num_trials = len(learning_output)
    phiEhats = [None] * total_num_trials
    phiAhats = [None] * total_num_trials
    rhoLTemp = [None] * total_num_trials

    # PhiE first
    plot_info.phi_type = 'energy'
    plot_info.phi_resolution = 1001
    phiE = []
    for ind in range(total_num_trials):
        phiEhats[ind] = learning_output[ind].Estimator.phiEhat[0][0]
        rhoLTemp[ind] = get_single_rhoLT(learning_output[ind].rhoLTemp.rhoLTA.rhoLTR, 1, 1)

    plot_info.display_interpolant = False
    rhoLT = []
    plot_interactions_and_rhos(fig, ax1, rhoLT, rhoLTemp, phiE, phiEhats, [], 1, 1, sys_info, plot_info)
    ax1.set_xlabel('$r$ (pairwise distance)', fontsize=plot_info.axis_font_size)
    ax1.set_ylabel('Interactions', fontsize=plot_info.axis_font_size)
    ax1.yaxis.set_major_formatter('${:.2g}$'.format)
    ax1.set_title('Energy Interaction ($\phi^E$)', fontsize=plot_info.axis_font_size)

    # PhiAhat second
    plot_info.phi_type = 'alignment'
    phiA = []
    for ind in range(total_num_trials):
        phiAhats[ind] = learning_output[ind].Estimator.phiAhat[0][0]

    plot_interactions_and_rhos(fig, ax2, rhoLT, rhoLTemp, phiA, phiAhats, [], 1, 1, sys_info, plot_info)
    ax2.set_xlabel('$r$ (pairwise distance)', fontsize=plot_info.axis_font_size)
    ax2.set_ylabel('Interactions', fontsize=plot_info.axis_font_size)
    ax2.yaxis.set_major_formatter('${:.2g}$'.format)
    ax2.set_title('Alignment Interaction ($\phi^A$)', fontsize=plot_info.axis_font_size)

    plt.tight_layout()

    # Save the figure
    plt.savefig(plot_info.plot_name + '_phi_MS.png', dpi=300)

    # Plot the joint distribution
    rho_fig = plt.figure('rhoLTemps: Marginals and Joint', figsize=(12, 6))
    plot_one_joint_distribution(rho_fig, learning_output[0].rhoLTemp, 1, plot_info)

    # Save the figure
    plt.savefig(plot_info.plot_name + '_rho_MS.png', dpi=300)

    plt.show()
