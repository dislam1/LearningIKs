import numpy as np
import matplotlib.pyplot as plt

def visualize_traj_1D(trajs, time_vec, sys_info, obs_info, plot_info):
    if sys_info['d'] != 1:
        raise ValueError('This routine is for 1D Visualization!!')

    # Prepare the window size
    if 'scrsz' in plot_info and plot_info['scrsz'] is not None:
        scrsz = plot_info['scrsz']
    else:
        scrsz = plt.gcf().get_size_inches() * plt.gcf().dpi

    # Prepare the figure window
    if 'for_larger_N' in plot_info and plot_info['for_larger_N'] and 'plot_noise' in plot_info and plot_info['plot_noise']:
        traj_fig, axes = plt.subplots(2, 2, figsize=(scrsz[0]*3/4, scrsz[1]*3/4))
    else:
        traj_fig, axes = plt.subplots(2, 2, figsize=(scrsz[0]*3/4, scrsz[1]*3/4))

    # Prepare the color for each type
    type_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if sys_info['K'] > 7:
        raise ValueError('The Coloring Scheme of trajectories only works for up to 7 types of agents!!')

    # Split the true trajectories (1)
    ind_1 = np.where(time_vec <= obs_info['T_L'])[0]
    ind_2 = np.where(time_vec >= obs_info['T_L'])[0]
    traj_1s, traj_2s = [], []
    time_1, time_2 = time_vec[ind_1], time_vec[ind_2]
    y_min, y_max = np.inf * np.ones(len(trajs)), -np.inf * np.ones(len(trajs))

    for traj in trajs:
        traj_1, traj_2 = traj[:, ind_1], traj[:, ind_2]
        traj_1s.append(traj_1)
        traj_2s.append(traj_2)
        y_min = np.minimum(y_min, np.min(traj))
        y_max = np.maximum(y_max, np.max(traj))

    y_range = y_max - y_min
    y_min = y_min - 0.1 * y_range
    y_max = y_max + 0.1 * y_range
    vline_y = np.linspace(y_min, y_max, obs_info['L'])
    vline_x = np.full_like(vline_y, obs_info['T_L'])
    x_min, x_max = np.min(time_vec), np.max(time_vec)

    for ind in range(4):
        row_ind = ind // 2
        col_ind = ind % 2
        ax = axes[row_ind, col_ind]

        traj_1, traj_2 = traj_1s[ind], traj_2s[ind]

        for k in range(sys_info['K']):
            agents_traj = traj_1[sys_info['type_info'] == k, :]
            ax.plot(time_1, agents_traj, linewidth=plot_info['traj_line_width'], color=type_colors[k])

            if k == 0:
                ax.hold(True)

        ax.plot(vline_x, vline_y, '-.k')

        for k in range(sys_info['K']):
            agents_traj = traj_2[sys_info['type_info'] == k, :]
            ax.plot(time_2, agents_traj, linewidth=plot_info['traj_line_width'], color=type_colors[k])

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.tick_params(axis='both', which='major', labelsize=plot_info['tick_font_size'], fontname=plot_info['tick_font_name'])

        if ind == 0 or ind == 2:
            ax.set_ylabel(r'$\mathbf{x}_i$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=10)
        if ind == 2 or ind == 3:
            ax.set_xlabel('Time $t$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=5)

        ax.hold(False)

    # Tighten them up
    for ax in axes.ravel():
        ax.get_figure().tight_layout(rect=[0, 0.03, 1, 0.95])

    if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
        plt.savefig(f"{plot_info['plot_name']}_traj_LN")
    else:
        plt.savefig(f"{plot_info['plot_name']}_traj")

    plt.show()
