import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def visualize_traj_2D_wnoise(traj_noise, traj_true, traj_hat, time_vec, sys_info, obs_info, plot_info):
    if sys_info['d'] != 2:
        raise ValueError('This routine is for 2D Visualization!!')

    # Prepare the window size
    if 'scrsz' in plot_info and plot_info['scrsz'] is not None:
        scrsz = plot_info['scrsz']
    else:
        scrsz = plt.gcf().get_size_inches() * plt.gcf().dpi

    traj_fig, ax = plt.subplots(figsize=(scrsz[0]*3/4, scrsz[1]*3/4))

    # Prepare the trajectories
    trajs = [traj_true, traj_hat, traj_noise]

    # Prepare the color items
    if 'for_PNAS' in plot_info and plot_info['for_PNAS']:
        color_output = construct_color_items(sys_info['K'], obs_info['T_L'], plot_info['coarse_time_vec'])
    else:
        color_output = construct_color_items(sys_info['K'], obs_info['T_L'], time_vec)

    cmap = color_output['cmap']
    c_vecs = color_output['c_vecs']
    clabels = color_output['clabels']
    cticks = color_output['cticks']

    # Split the trajectories
    X_c1s, X_c2s = [], []
    x_min, x_max = np.inf * np.ones(3), -np.inf * np.ones(3)
    y_min, y_max = np.inf * np.ones(3), -np.inf * np.ones(3)

    for ind in range(3):
        traj = trajs[ind]
        X_c1, X_c2 = traj[::2, :], traj[1::2, :]
        x_min[ind] = np.min(X_c1)
        x_max[ind] = np.max(X_c1)
        y_min[ind] = np.min(X_c2)
        y_max[ind] = np.max(X_c2)
        X_c1s.append(X_c1)
        X_c2s.append(X_c2)

    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_min = np.min(y_min)
    y_max = np.max(y_max)

    l_handles = []

    line_styles = ['-', '-.', '--']
    T_loc = np.where(time_vec == obs_info['T_L'])[0][0]
    missing_len = len(time_vec) - T_loc

    for ind in range(3):
        X_c1 = X_c1s[ind]
        X_c2 = X_c2s[ind]

        for k in range(sys_info['K']):
            agents_Ck = np.where(sys_info['type_info'] == k)[0]
            N_k = len(agents_Ck)

            for agent_ind in range(N_k):
                agent = agents_Ck[agent_ind]

                if ind == 2:
                    c1_at_t = np.concatenate([X_c1[agent, :], np.full(missing_len, np.nan)])
                    c2_at_t = np.concatenate([X_c2[agent, :], np.full(missing_len, np.nan)])
                else:
                    c1_at_t = X_c1[agent, :]
                    c2_at_t = X_c2[agent, :]

                if 'for_PNAS' in plot_info and plot_info['for_PNAS']:
                    if ind < 2:
                        ax.plot(c1_at_t, c2_at_t, linewidth=plot_info['traj_line_width'], color='k', linestyle=line_styles[ind])
                        if k == 0 and agent_ind == 0:
                            ax.hold(True)

                        dyn_markers = plot_info['dyn_markers'][ind]
                        m_C1 = dyn_markers[::2, :]
                        m_C2 = dyn_markers[1::2, :]
                        mC1_at_t = m_C1[agent, :]
                        mC2_at_t = m_C2[agent, :]
                        ax.scatter(mC1_at_t, mC2_at_t, s=plot_info['marker_size'], c=c_vecs[k], marker=plot_info['marker_style'][k])
                    else:
                        s_handle = ax.scatter(c1_at_t, c2_at_t, s=plot_info['marker_size'], c='k', marker='s', alpha=0.2)
                else:
                    p_handle = ax.fill_between(c1_at_t, c2_at_t, np.nan, color=c_vecs[k], edgecolor='none', linewidth=plot_info['traj_line_width'])
                    if ind == 2:
                        p_handle.set_alpha(0.2)
                    if k == 0 and agent_ind == 0:
                        ax.hold(True)

        if ind < 2:
            l_handle = ax.plot(np.nan, np.nan, 'k' + line_styles[ind])[0]
        else:
            l_handle = s_handle

        l_handles.append(l_handle)

    ax.set_xlim([x_min, x_max + delta])
    ax.set_ylim([y_min, y_max])

    ax.tick_params(axis='both', which='major', labelsize=plot_info['tick_font_size'], fontname=plot_info['tick_font_name'])

    cmap = plt.get_cmap(cmap)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=cticks)
    cbar.set_ticklabels(clabels)

    ax.set_ylabel(r'Coord. $2$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=10)
    ax.set_xlabel(r'Coord. $1$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=5)

    leg_handle = ax.legend(l_handles, ['$\mathbf{x}_i(t)$', '$\hat\mathbf{x}_i(t)$', '$\mathbf{x}_i^\epsilon(t)$'],
                           loc='upper right', fontsize=plot_info['legend_font_size'])

    ax.get_figure().tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f"{plot_info['plot_name']}_traj_noise")
    plt.show()
