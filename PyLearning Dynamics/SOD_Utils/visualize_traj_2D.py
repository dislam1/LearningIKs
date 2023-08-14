import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.collections import PatchCollection

from SOD_Utils.get_colored_line import get_colored_line
from SOD_Utils.construct_color_items import construct_color_items

def visualize_traj_2D(trajs, time_vec, sys_info, obs_info, plot_info):
    if sys_info['d'] != 2:
        raise ValueError('This routine is for 2D Visualization!!')

    # Prepare the window size
    if 'scrsz' in plot_info and plot_info['scrsz'] is not None:
        scrsz = plot_info['scrsz']
    else:
        scrsz = plt.gcf().get_size_inches() * plt.gcf().dpi

    # Prepare the figure window
    if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
        traj_fig, axes = plt.subplots(2, 2, figsize=(14,9))
    else:
        traj_fig, axes = plt.subplots(2, 2, figsize=(14,9))

    #Construct color items
    if plot_info['for_PNAS']:
        color_output = construct_color_items(sys_info['K'], obs_info['T_L'], plot_info['coarse_time_vec'])
    else:
        color_output = construct_color_items(sys_info['K'], obs_info['T_L'], time_vec)

    cmap = color_output['cmap']
    #c_vecs = plot_info.get('c_vecs', np.linspace(0, 1, sys_info['K']))
    c_vecs = color_output['c_vecs']
    clabels = color_output['clabels']
    cticks = color_output['cticks']

    # Split the trajectories
    X_c1s, X_c2s = [None] * 4, [None] * 4
    x_min, x_max = np.zeros(4), np.zeros(4)
    y_min, y_max = np.zeros(4), np.zeros(4)

    for ind, traj in enumerate(trajs):
       
        end1 = traj.shape[0]
        X_c1, X_c2 = traj[0:end1 -1:2, :], traj[1:end1:2, :]
        X_c1s[ind] = X_c1
        X_c2s[ind] = X_c2
        x_min[ind]      = min(X_c1.ravel())
        x_max[ind]      = max(X_c1.ravel())
        y_min[ind]      = min(X_c2.ravel())
        y_max[ind]      = max(X_c2.ravel())
    

    x_min = np.min(x_min)
    x_max = np.max(x_max)
    y_min = np.min(y_min)
    y_max = np.max(y_max)

    T_loc = np.where(time_vec == obs_info['T_L'])[0][0]

    for ind in range(4):
        #plt.subplots(2, 2)
        row_ind = ind // 2
        col_ind = ind % 2
        ax = axes[row_ind, col_ind]
        #ax.set_xlim([x_min - 0.5, x_max+0.5])
        #ax.set_ylim([y_min - 0.5, y_max + 0.5])

        X_c1, X_c2 = X_c1s[ind], X_c2s[ind]
        c1_at_t = []
        c2_at_t = []
        for k in range(sys_info['K']):
            agents_Ck = np.where(np.array(sys_info['type_info']) == k+1)[0]
            N_k = len(agents_Ck)
            #plot = ax.scatter([], [])
            for agent_ind in range(N_k):
                agent = agents_Ck[agent_ind]
                c1_at_t.append(X_c1[agent, :])
                c2_at_t.append(X_c2[agent, :])
               
                
        
        cmap = plt.cm.viridis
        polygons = [Polygon(np.column_stack((x_row,y_row))) for x_row, y_row in zip(c1_at_t,c2_at_t)]
        collection = PatchCollection(polygons, cmap = cmap, alpha = 0.5)
        collection.set_array(c_vecs[k])
        collection.set_cmap(cmap)
        ax.add_collection(collection)
                    
        ax.scatter(c1_at_t,c2_at_t)
        #get_colored_line(c1_at_t, c2_at_t, c_vecs[k],ax)
        ax.scatter(X_c1[:, T_loc], X_c2[:, T_loc], s=plot_info['T_L_marker_size'], c='k', marker='o')

        ax.set_xlim([x_min-0.5, x_max+0.5])
        ax.set_ylim([y_min - 0.5, y_max+0.5])

        ax.tick_params(axis='both', which='major', labelsize=plot_info['tick_font_size'])

        if ind == 0:
            xticks = ax.get_xticks()
            delta = xticks[1] - xticks[0]
        elif ind % 2 == 1:
            ax.set_xlim([x_min - delta, x_max + delta])

        if ind % 2 == 0:
            ax.set_ylabel(r'Coord. $2$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=10)

        if ind == 2 or ind == 3:
            ax.set_xlabel(r'Coord. $1$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'], labelpad=5)

        if 'for_PNAS' in plot_info and plot_info['for_PNAS']:
            # The PNAS specific settings
            dyn_markers = plot_info['dyn_markers'][ind]
            m_C1, m_C2 = dyn_markers[::2, :], dyn_markers[1::2, :]

            for k in range(sys_info['K']):
                agents_Ck = np.where(sys_info['type_info'] == k)[0]
                N_k = len(agents_Ck)

                for agent_ind in range(N_k):
                    agent = agents_Ck[agent_ind]
                    mC1_at_t = m_C1[agent, :]
                    mC2_at_t = m_C2[agent, :]
                    ax.scatter(mC1_at_t, mC2_at_t, s=plot_info['marker_size'], c=c_vecs[k], marker=plot_info['marker_style'][k])

        leg_name = r'$\mathbf{x}_i(t)$' if ind % 2 == 1 else r'$\hat\mathbf{x}_i(t)$'
        leg_handle = ax.legend([Patch(color='k', linestyle=plot_info['line_styles'][0])], [leg_name],
                               loc=plot_info.get('leg_loc', 'best'), fontsize=plot_info['legend_font_size'])

        #cmap = plt.get_cmap(cmap)
        #if ind % 2 == 1:
            #cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, ticks=cticks)
            #cbar.set_ticklabels(clabels)

    #for ax in axes.ravel():
    #    ax.get_figure().tight_layout(rect=[0, 0.03, 1, 0.95])
    traj_fig.tight_layout()

    if 'for_larger_N' in plot_info and plot_info['for_larger_N']:
        plt.savefig(f"{plot_info['plot_name']}_traj_LN")
    else:
        plt.savefig(f"{plot_info['plot_name']}_traj")

    #plt.show()
    if sys_info['debug_mode']:
        plt.show()
    else:
        plt.ion()
    #plt.show(block = False)
