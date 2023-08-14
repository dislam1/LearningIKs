import matplotlib.pyplot as plt
import numpy as np

def plot_one_joint_distribution(win_handler, jrhoLT, K, plot_info):
    # Set the current figure to the specified window handler
    plt.figure(win_handler)
    color_true = [0.8500, 0.3250, 0.0980]
    
    handleAxes = []
    
    for k1 in range(1, K + 1):
        for k2 in range(1, K + 1):
            # Plots for rho^L_T, rho^L_{T, r} first
            plt.subplot(3 * K, K, (k1 - 1) * 3 * K + k2)
            range_val = jrhoLT['rhoLTA']['rhoLTR']['supp'][k1 - 1, k2 - 1][0, :]
            edges = jrhoLT['rhoLTA']['rhoLTR']['histedges'][k1 - 1, k2 - 1]
            edges_idxs = np.where((range_val[0] <= edges) & (edges <= range_val[1]))[0]
            histdata, edges = downsampleHistCounts(jrhoLT['rhoLTA']['rhoLTR']['hist'][k1 - 1, k2 - 1][edges_idxs[:-1]], 
                                                   edges[edges_idxs], np.sqrt(len(edges_idxs))/2)
            centers = (edges[:-1] + edges[1:]) / 2
            plt.plot(centers, histdata, 'k', linewidth=1, color=color_true)
            plt.fill_between(centers, 0, histdata, color=color_true/2, alpha=0.1)
            plt.axis('tight')
            tmpmax = plot_info['rhotscalingdownfactor'] * max(histdata)
            if tmpmax == 0:
                tmpmax = 1
            if np.isnan(tmpmax):
                tmpmax = 1
            if abs(range_val[1] - range_val[0]) < 1.0e-12:
                range_val[1] = range_val[0] + 1
            plt.axis([range_val[0], range_val[1], 0, tmpmax])
            Escale = int(np.log10(tmpmax)) if tmpmax != 1 else 1
            plt.gca().yaxis.set_exponent(Escale)
            plt.gca().yaxis.set_major_formatter('{:.2g}'.format)
            plt.gca().tick_params(axis='both', labelsize=plot_info['tick_font_size'])
            plt.xlabel('$r$ (pairwise distance)', fontsize=plot_info['axis_font_size'])
            ylabel_str = '$\\rho^L_{T, r}$' if K == 1 else f'$\\rho_{{T, r}}^{{T, {k1}{k2}}}$'
            plt.ylabel(ylabel_str, fontsize=plot_info['axis_font_size'])
            handleAxes.append(plt.gca())
            
            # rho^L_{T, \dot{r}} or \rho^L_{T, \xi} second
            plt.subplot(3 * K, K, (k1 - 1) * 3 * K + k2 + K)
            range_val = jrhoLT['rhoLTA']['rhoLTDR']['supp'][k1 - 1, k2 - 1][0, :]
            edges = jrhoLT['rhoLTA']['rhoLTDR']['histedges'][k1 - 1, k2 - 1]
            edges_idxs = np.where((range_val[0] <= edges) & (edges <= range_val[1]))[0]
            histdata, edges = downsampleHistCounts(jrhoLT['rhoLTA']['rhoLTDR']['hist'][k1 - 1, k2 - 1][edges_idxs[:-1]], 
                                                   edges[edges_idxs], np.sqrt(len(edges_idxs))/2)
            centers = (edges[:-1] + edges[1:]) / 2
            plt.plot(centers, histdata, 'k', linewidth=1, color=color_true)
            plt.fill_between(centers, 0, histdata, color=color_true/2, alpha=0.1)
            plt.axis('tight')
            tmpmax = plot_info['rhotscalingdownfactor'] * max(histdata)
            if tmpmax == 0:
                tmpmax = 1
            if np.isnan(tmpmax):
                tmpmax = 1
            if abs(range_val[1] - range_val[0]) < 1.0e-12:
                range_val[1] = range_val[0] + 1
            plt.axis([range_val[0], range_val[1], 0, tmpmax])
            Escale = int(np.log10(tmpmax)) if tmpmax != 1 else 1
            plt.gca().yaxis.set_exponent(Escale)
            plt.gca().yaxis.set_major_formatter('{:.2g}'.format)
            plt.gca().tick_params(axis='both', labelsize=plot_info['tick_font_size'])
            plt.xlabel('$\\dot{r}$ (pairwise speed)', fontsize=plot_info['axis_font_size'])
            ylabel_str = '$rho^L_{T, dot[r]$' if K == 1 else f'$\\rho_{{T, \\dot[r]^{{T, {k1}{k2}}}$'
            plt.ylabel(ylabel_str, fontsize=plot_info['axis_font_size'])
            handleAxes.append(plt.gca())
            
            # Joint \rho^L_{T, r, \dot{r}} or \rho^L_{T, r, \xi}
            plt.subplot(3 * K, K, (k1 - 1) * 3 * K + k2 + 2 * K)
            x_range = jrhoLT['rhoLTA']['supp'][k1 - 1, k2 - 1][0, :]
            x_edges = jrhoLT['rhoLTA']['histedges'][k1 - 1, k2 - 1][0, :]
            x_edges_idxs = np.where((x_range[0] <= x_edges) & (x_edges <= x_range[1]))[0]
            num_xeidxs = len(x_edges_idxs)
            y_range = jrhoLT['rhoLTA']['supp'][k1 - 1, k2 - 1][1, :]
            y_edges = jrhoLT['rhoLTA']['histedges'][k1 - 1, k2 - 1][1, :]
            y_edges_idxs = np.where((y_range[0] <= y_edges) & (y_edges <= y_range[1]))[0]
            num_yeidxs = len(y_edges_idxs)
            histdata, x_edges, y_edges = downsampleHist2D(jrhoLT['rhoLTA']['hist'][k1 - 1, k2 - 1][x_edges_idxs[:-1], y_edges_idxs[:-1]], 
                                                         x_edges[x_edges_idxs], np.sqrt(num_xeidxs)/2, 
                                                         y_edges[y_edges_idxs], np.sqrt(num_yeidxs)/2)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            pc_handle = plt.pcolormesh(x_centers, y_centers, histdata.T, shading='auto')
            pc_handle.set_edgecolor('none')
            plt.axis([x_range[0], x_range[1], y_range[0], y_range[1]])
            plt.gca().tick_params(axis='both', labelsize=plot_info['tick_font_size'])
            plt.xlabel('$r$ (pairwise distance)', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'])
            plt.ylabel('$\\dot{r}$', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'])
            plt.colorbar(label='Counts', ax=plt.gca())
            handleAxes.append(plt.gca())
            
    # Adjust the figure layout
    tightFigaroundAxes(handleAxes)
    plt.show()
