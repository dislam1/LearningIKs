import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualize_traj_3D(trajs, time_vec, sys_info, obs_info, plot_info):
    if sys_info['d'] != 3:
        raise ValueError('This routine is for 3D Visualization!!')

    # Prepare the window size
    if 'scrsz' in plot_info and plot_info['scrsz'] is not None:
        scrsz = plot_info['scrsz']
    else:
        scrsz = plt.gcf().get_size_inches() * plt.gcf().dpi

    if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
        traj_fig, axs = plt.subplots(2, 2, figsize=(scrsz[0]*3/4, scrsz[1]*3/4))
    else:
        traj_fig, axs = plt.subplots(2, 2, figsize=(scrsz[0]*3/4, scrsz[1]*3/4))

    # Prepare the color items
    color_output = construct_color_items(sys_info['K'], obs_info['T_L'], time_vec)
    cmap = color_output['cmap']
    c_vecs = color_output['c_vecs']
    clabels = color_output['clabels']
    cticks = color_output['cticks']

    # Split the trajectories
    X_c1s, X_c2s, X_c3s = [], [], []
    x_min, x_max = np.inf * np.ones(4), -np.inf * np.ones(4)
    y_min, y_max = np.inf * np.ones(4), -np.inf * np.ones(4)
    z_min, z_max = np.inf * np.ones(4), -np.inf * np.ones(4)

    for ind in range(4):
        traj = trajs[ind]
        X_c1, X_c2, X_c3 = traj[::3, :], traj[1::3, :], traj[2::3, :]
        x_min[ind] = np.min(X_c1)
        x_max[ind] = np.max(X_c1)
        y_min[ind] = np.min(X_c2)
        y_max[ind] = np.max(X_c2)
        z_min[ind] = np.min(X_c3)
        z_max[ind] = np.max(X_c3)
        X_c1s.append(X_c1)
        X_c2s.append(X_c2)
        X_c3s.append(X_c3)

    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_min = np.min(y_min)
    y_max = np.max(y_max)
    z_min = np.min(z_min)
    z_max = np.max(z_max)

    T_loc = np.where(time_vec == obs_info['T_L'])[0][0]

    for ind in range(4):
        ax = axs[ind//2, ind%2]

        X_c1 = X_c1s[ind]
        X_c2 = X_c2s[ind]
        X_c3 = X_c3s[ind]

        for k in range(sys_info['K']):
            agents_Ck = np.where(sys_info['type_info'] == k)[0]
            N_k = len(agents_Ck)

            for agent_ind in range(N_k):
                agent = agents_Ck[agent_ind]
                c1_at_t = X_c1[agent, :]
                c2_at_t = X_c2[agent, :]
                c3_at_t = X_c3[agent, :]

                p_handle = ax.plot(c1_at_t, c2_at_t, c3_at_t, c=c_vecs[k], linestyle='-', linewidth=plot_info['traj_line_width'])
                if k == 0 and agent_ind == 0:
                    ax.hold(True)

        ax.plot(X_c1[:, T_loc], X_c2[:, T_loc], X_c3[:, T_loc], 'o', markersize=plot_info['T_L_marker_size'], color='k')

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.tick_params(axis='both', which='major', labelsize=plot_info['tick_font_size'], fontname=plot_info['tick_font_name'])

        ax.set_xlabel(r'Coord. $1$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=5)
        ax.set_ylabel(r'Coord. $2$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=5)
        ax.set_zlabel(r'Coord. $3$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=5)

        colormap = plt.cm.ScalarMappable(cmap=cmap)
        cbar = plt.colorbar(colormap, ax=ax)
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(clabels)
        cbar.ax.tick_params(labelsize=plot_info['colorbar_font_size'])

    traj_fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
        plt.savefig(f"{plot_info['plot_name']}_traj_LN")
    else:
        plt.savefig(f"{plot_info['plot_name']}_traj")

    plt.show()
