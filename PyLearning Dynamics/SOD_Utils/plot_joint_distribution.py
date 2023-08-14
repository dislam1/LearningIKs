import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_joint_distribution(win_handler, jrhoLT, jrhoLTemp, K, plot_info):
    # Function to downsample histogram counts
    def downsample_hist_counts(histdata, edges, factor):
        return histdata[::factor], edges[::factor]

    # Function to downsample 2D histogram counts
    def downsample_hist2D(histdata, x_edges, x_factor, y_edges, y_factor):
        return histdata[::x_factor, ::y_factor], x_edges[::x_factor], y_edges[::y_factor]

    # Create a new figure and set the current figure
    plt.figure(win_handler.number)
    
    # Colors for the true and empirical plots
    color_true = [0.8500, 0.3250, 0.0980]
    color_emp = [0.8500, 0.3250, 0.0980]

    # Go through each (k1, k2) pair
    for k1 in range(1, K + 1):
        for k2 in range(1, K + 1):
            # Plots for the true joint distribution, rho^L_T, rho^L_{T, r} first
            plt.subplot(3 * K, 2 * K, (k1 - 1) * 2 * K + (k2 - 1) * 2 + 1)
            rhoLTR = jrhoLT["rhoLTR"]
            range_r = jrhoLTemp["supp"][k1 - 1][k2 - 1][0]
            edges = rhoLTR["histedges"][k1 - 1][k2 - 1]
            edges_idxs = np.logical_and(range_r[0] <= edges, edges <= range_r[1])
            histdata, edges = downsample_hist_counts(rhoLTR["hist"][k1 - 1][k2 - 1][edges_idxs[:-1]], edges[edges_idxs], np.sqrt(len(edges_idxs)) // 2)
            centers = (edges[:-1] + edges[1:]) / 2
            histHandle = plt.plot(centers, histdata, 'k', linewidth=1, color=color_true)
            plt.fill_between(centers, np.concatenate(([0], histdata, [0])), color=color_true, alpha=0.1)
            plt.axis('tight')
            plt.xlabel('$r$ (pairwise distance)', fontsize=plot_info["axis_font_size"])
            if K == 1:
                plt.ylabel('$\\rho^L_{T, r}$', fontsize=plot_info["axis_font_size"])
            else:
                plt.ylabel(f'$\\rho_{{T, r}}^{{T, {k1}{k2}}}$', fontsize=plot_info["axis_font_size"])

            # \rho^L_{T, \dot{r}} or \rho^L_{T, \xi} second
            plt.subplot(3 * K, 2 * K, (k1 - 1) * 2 * K + (k2 - 1 + K) * 2 + 1)
            if plot_info["phi_type"] == 'alignment':
                rhoLTo = jrhoLT["rhoLTDR"]
            elif plot_info["phi_type"] == 'xi':
                rhoLTo = jrhoLT["mrhoLTXi"]
            range_rdot = jrhoLTemp["supp"][k1 - 1][k2 - 1][1]
            edges = rhoLTo["histedges"][k1 - 1][k2 - 1]
            edges_idxs = np.logical_and(range_rdot[0] <= edges, edges <= range_rdot[1])
            histdata, edges = downsample_hist_counts(rhoLTo["hist"][k1 - 1][k2 - 1][edges_idxs[:-1]], edges[edges_idxs], np.sqrt(len(edges_idxs)) // 2)
            centers = (edges[:-1] + edges[1:]) / 2
            histHandle = plt.plot(centers, histdata, 'k', linewidth=1, color=color_true)
            plt.fill_between(centers, np.concatenate(([0], histdata, [0])), color=color_true, alpha=0.1)
            plt.axis('tight')
            plt.xlabel('$\\dot{r}$ (pairwise speed)', fontsize=plot_info["axis_font_size"])
            if plot_info["phi_type"] == 'alignment':
                if K == 1:
                    plt.ylabel('$\\rho^L_{T, \\dot{r}}$', fontsize=plot_info["axis_font_size"])
                else:
                    plt.ylabel(f'$\\rho_{{T, \\dot[r]^{{T, {k1}{k2}}}$', fontsize=plot_info["axis_font_size"])
            elif plot_info["phi_type"] == 'xi':
                if K == 1:
                    plt.ylabel('$\\rho^L_{T, \\xi}$', fontsize=plot_info["axis_font_size"])
                else:
                    plt.ylabel(f'$\\rho_{{T, \\xi}}^{{T, {k1}{k2}}}$', fontsize=plot_info["axis_font_size"])

            # Joint \rho^L_{T, r, \dot{r}} or \rho^L_{T, r, \xi}
            plt.subplot(3 * K, 2 * K, (k1 - 1) * 2 * K + (k2 - 1 + 2 * K) * 2 + 1)
            x_range = jrhoLTemp["supp"][k1 - 1][k2 - 1][0]
            x_edges = jrhoLT["histedges"][k1 - 1][k2 - 1][0]
            x_edges_idxs = np.logical_and(x_range[0] <= x_edges, x_edges <= x_range[1])
            num_xeidxs = np.sum(x_edges_idxs)
            y_range = jrhoLTemp["supp"][k1 - 1][k2 - 1][1]
            y_edges = jrhoLT["histedges"][k1 - 1][k2 - 1][1]
            y_edges_idxs = np.logical_and(y_range[0] <= y_edges, y_edges <= y_range[1])
            num_yeidxs = np.sum(y_edges_idxs)
            histdata, x_edges, y_edges = downsample_hist2D(jrhoLT["hist"][k1 - 1][k2 - 1][x_edges_idxs[:-1]][:, y_edges_idxs[:-1]], x_edges[x_edges_idxs], np.sqrt(num_xeidxs) // 2, y_edges[y_edges_idxs], np.sqrt(num_yeidxs) // 2)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            pc_handle = plt.pcolormesh(x_centers, y_centers, histdata.T, norm=LogNorm(), cmap='jet')
            pc_handle.set_edgecolor('face')
            plt.axis('tight')
            plt.xlabel('$r$ (pairwise distance)', fontsize=plot_info["axis_font_size"])
            if plot_info["phi_type"] == 'alignment':
                plt.ylabel('$\\dot{r}$ (pairwise speed)', fontsize=plot_info["axis_font_size"])
            elif plot_info["phi_type"] == 'xi':
                plt.ylabel('$\\xi$ (pairwise $\\xi_{i, i\'}$)', fontsize=plot_info["axis_font_size"])
            plt.colorbar(pc_handle)

    plt.show()


# Example usage (replace the input arguments with your data)
# plot_joint_distribution(win_handler, jrhoLT, jrhoLTemp, K, plot_info)
