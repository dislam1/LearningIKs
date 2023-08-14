import numpy as np
import matplotlib.pyplot as plt


def plot_phiA_and_phiXi_together(win_handle, learningOutput, sys_info, obs_info, plot_info):
    # Function to get single rhoLT for a (k1, k2) pair
    def get_single_rhoLT(rhoLT, k1, k2):
        return rhoLT[k1 - 1][k2 - 1]

    # Function to get the range from rhos
    def get_range_from_rhos(rhoLTemp):
        rmin = np.min([np.min(rho) for rho in rhoLTemp])
        rmax = np.max([np.max(rho) for rho in rhoLTemp])
        return rmin, rmax

    # Function to plot rhos
    def plot_rhos(axis_handle, range_vals, rhoLT, rhoLTemp, k1, k2, sys_info, plot_info):
        rmin, rmax = range_vals
        rhoPlotHandles = []
        rhoPlotNames = []
        for idx in range(len(rhoLTemp)):
            rhoTemp = rhoLTemp[idx]
            rhoPlot, = axis_handle.plot(rhoTemp, np.linspace(rmin, rmax, len(rhoTemp)), plot_info.linestyles[idx], linewidth=2)
            rhoPlotHandles.append(rhoPlot)
            rhoPlotNames.append(f'Trial {idx+1}')
        rhoLTPlot, = axis_handle.plot(rhoLT, np.linspace(rmin, rmax, len(rhoLT)), 'k-', linewidth=2)
        rhoPlotHandles.append(rhoLTPlot)
        rhoPlotNames.append('Empirical')
        axis_handle.set_yticklabels([])
        axis_handle.set_xlim(0, 1)
        axis_handle.set_ylim(rmin, rmax)
        axis_handle.invert_yaxis()
        axis_handle.grid(True)
        return rhoPlotHandles, rhoPlotNames

    # Function to plot interactions and rhos
    def plot_interactions_and_rhos(win_handle, axis_handle, rhoLT, rhoLTemp, phi, phihats, phihatsmooths, k1, k2, sys_info, plot_info):
        # Plot phi
        phi_handle, = axis_handle.plot(phi, 'b-', linewidth=2, label='True')
        phihats_handles = []
        for idx in range(len(phihats)):
            phihat = phihats[idx]
            phihat_handle, = axis_handle.plot(phihat, plot_info.linestyles[idx], linewidth=2, label=f'Trial {idx+1}')
            phihats_handles.append(phihat_handle)
        axis_handle.set_xlim(0, len(phi))
        axis_handle.set_ylim(0, 1)
        axis_handle.grid(True)
        # Plot rhos
        range_vals = get_range_from_rhos(rhoLTemp)
        rhoPlotHandles, rhoPlotNames = plot_rhos(axis_handle, range_vals, rhoLT, rhoLTemp, k1, k2, sys_info, plot_info)
        if k1 == sys_info.K:
            axis_handle.set_xlabel('$r$ (pairwise distance)', fontsize=plot_info.axis_font_size, fontname=plot_info.axis_font_name)
        if plot_info.showplottitles:
            title_str = f'N = {sys_info.N}, T = {obs_info.T_L}, M = {obs_info.M}, L = {obs_info.L}, n = {basis_info.n[k1-1, k2-1]}'
            axis_handle.set_title(title_str, fontsize=plot_info.title_font_size, fontweight='bold')
            axis_handle.set_position(axis_handle.get_position() * [1, 0.95, 1])
        if not plot_info.phi_type == 'energy':
            ax = plt.subplot(sys_info.K, 2 * sys_info.K, (k1 - 1) * 2 * sys_info.K + 2 * k2)
            ax.yaxis.tick_right()
            rhoLT = get_single_rhoLT(rhoLTDR, k1, k2)
            range_vals = get_range_from_rhos(rhoLTemp)
            [rhoPlotHandles, rhoPlotNames] = plot_rhos(ax, range_vals, rhoLT, rhoLTemp, k1, k2, sys_info, plot_info)
            ax.legend(rhoPlotHandles, rhoPlotNames, loc='upper right', fontsize=plot_info.legend_font_size, frameon=True, framealpha=1)
            if k1 == sys_info.K:
                if plot_info.phi_type == 'alignment':
                    xlabel_handle = plt.xlabel('$\dot{r}$ (pairwise speed)', fontsize=plot_info.axis_font_size)
                elif plot_info.phi_type == 'xi':
                    xlabel_handle = plt.xlabel('$\\xi$ (pairwise $\\xi_{i, i\'}$)', fontsize=plot_info.axis_font_size)
                ax.xaxis.set_label_coords(0.5, -0.15)
                ax.set_position(ax.get_position() * [1, 1, 1, 0.95])
                ax.yaxis.set_tick_params(width=0.5)
                ax.xaxis.set_tick_params(width=0.5)
                ax.yaxis.tick_right()
                ax.set_xlim(0, 1)
        axis_handle.yaxis.set_major_formatter('{:.2g}'.format)
        axis_handle.tick_params(axis='both', which='major', width=0.5)
        axis_handle.tick_params(axis='both', which='minor', width=0.5)

    total_num_trials = len(learningOutput)
    rhoLTR = obs_info["rhoLT"]["rhoLTA"]["rhoLTR"]
    rhoLTRemp = [{} for _ in range(total_num_trials)]
    rhoLTDR = obs_info["rhoLT"]["rhoLTA"]["rhoLTDR"]
    rhoLTDRemp = [{} for _ in range(total_num_trials)]
    rhoLTXi = obs_info["rhoLT"]["rhoLTXi"]["mrhoLTXi"]
    rhoLTXiemp = [{} for _ in range(total_num_trials)]
    rhoLTemp = [{} for _ in range(total_num_trials)]

    all_phiAhat = [{} for _ in range(total_num_trials)]
    all_phiXihat = [{} for _ in range(total_num_trials)]
    all_phiAhatsmooth = [{} for _ in range(total_num_trials)]
    all_phiXihatsmooth = [{} for _ in range(total_num_trials)]

    for idx in range(total_num_trials):
        rhoLTRemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTA"]["rhoLTR"]
        rhoLTDRemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTA"]["rhoLTDR"]
        rhoLTXiemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTXi"]["mrhoLTXi"]
        all_phiAhat[idx] = learningOutput[idx]["Estimator"]["phiAhat"][0][0]
        all_phiAhatsmooth[idx] = learningOutput[idx]["Estimator"]["phiAhatsmooth"][0][0]
        all_phiXihat[idx] = learningOutput[idx]["Estimator"]["phiXihat"][0][0]
        all_phiXihatsmooth[idx] = learningOutput[idx]["Estimator"]["phiXihatsmooth"][0][0]

    plot_info["phi_resolution"] = 1001
    phiA = sys_info["phiA"]
    phiXi = sys_info["phiXi"]

    plt.figure(win_handle)
    for plot_ind in range(1, 3):
        axis_handle = plt.subplot(2, 2, (plot_ind - 1) * 2 + 1)
        if plot_ind == 1:
            phi = phiA[0][0]
            phihats = all_phiAhat
            phihatsmooths = all_phiAhatsmooth
            plot_info["phi_type"] = 'alignment'
        else:
            phi = phiXi[0][0]
            phihats = all_phiXihat
            phihatsmooths = all_phiXihatsmooth
            plot_info["phi_type"] = 'xi'
        rhoLT = get_single_rhoLT(rhoLTR, 1, 1)
        for ind in range(len(learningOutput)):
            rhoLTemp[ind] = get_single_rhoLT(rhoLTRemp[ind], 1, 1)
        plot_interactions_and_rhos(win_handle, axis_handle, rhoLT, rhoLTemp, phi, phihats, phihatsmooths, 1, 1, sys_info, plot_info)
        xlabel_str = '$r$ (pairwise distance)'
        plt.xlabel(xlabel_str, fontsize=plot_info["axis_font_size"], fontname=plot_info["axis_font_name"])

        handleAxes[plot_ind - 1][0] = axis_handle

        axis_handle = plt.subplot(2, 2, (plot_ind - 1) * 2 + 2)
        plt.yyaxis('right')
        if plot_ind == 1:
            rhoLT = get_single_rhoLT(rhoLTDR, 1, 1)
            for ind in range(len(learningOutput)):
                rhoLTemp[ind] = get_single_rhoLT(rhoLTDRemp[ind], 1, 1)
        else:
            rhoLT = get_single_rhoLT(rhoLTXi, 1, 1)
            for ind in range(len(learningOutput)):
                rhoLTemp[ind] = get_single_rhoLT(rhoLTXiemp[ind], 1, 1)

        range_vals = get_range_from_rhos(rhoLTemp)
        rhoPlotHandles, rhoPlotNames = plot_rhos(axis_handle, range_vals, rhoLT, rhoLTemp, 1, 1, sys_info, plot_info)
        legend_handle = plt.legend(rhoPlotHandles, rhoPlotNames, loc='upper right', fontsize=plot_info["legend_font_size"], frameon=True, framealpha=1)
        if plot_ind == 1:
            xlabel_str = '$\dot{r}$ (pairwise speed)'
        else:
            xlabel_str = '$\\xi$ (pairwise $\\xi_{i, i\'}$)'

        plt.xlabel(xlabel_str, fontsize=plot_info["axis_font_size"], fontname=plot_info["axis_font_name"])
        plt.ylabel('')
        plt.twinx()
        handleAxes[plot_ind - 1][1] = plt.gca()

    for ind in range(2):
        handleAxes[ind][0].yaxis.set_major_formatter('{:.2g}'.format)

    plt.tight_layout()
    plt.show()
