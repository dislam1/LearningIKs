import numpy as np
import matplotlib.pyplot as plt
from SOD_Utils.get_single_rhoLT import get_single_rhoLT
from SOD_Utils.plot_interactions_and_rhos import plot_interactions_and_rhos
from  SOD_Utils.get_range_from_rhos import  get_range_from_rhos
from SOD_Utils.plot_rhos  import plot_rhos

def plot_one_type_of_phis(win_handler, phi, all_phihat, all_phihatsmooth, learningOutput, basis_info, sys_info, obs_info, plot_info):
    # Function to get single rhoLT for a (k1, k2) pair
    
   
    total_num_trials = len(learningOutput)
    if plot_info['phi_type'] == 'energy':
        rhoLTR = obs_info["rhoLT"]["rhoLTE"]
    elif plot_info['phi_type'] == 'alignment':
        rhoLTR = obs_info["rhoLT"]["rhoLTA"]["rhoLTR"]
        rhoLTA = obs_info["rhoLT"]["rhoLTA"]["rhoLTDR"]
        rhoLTAemp = [{} for _ in range(total_num_trials)]
    elif plot_info['phi_type'] == 'xi':
        rhoLTR = obs_info["rhoLT"]["rhoLTXi"]["rhoLTR"]
        rhoLTA = obs_info["rhoLT"]["rhoLTXi"]["mrhoLTXi"]
        rhoLTAemp = [{} for _ in range(total_num_trials)]
    else:
        return

    rhoLTRemp = [None]*total_num_trials
    rhoLTemp = [None]*total_num_trials
    phihats = [None]*total_num_trials
    phihatsmooths = [None]*total_num_trials

    for idx in range(total_num_trials):
        if plot_info['phi_type'] == 'energy':
            rhoLTRemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTE"]
            handleAxes = np.empty(sys_info['K'], dtype=object)
        elif plot_info['phi_type'] == 'alignment':
            rhoLTRemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTA"]["rhoLTR"]
            rhoLTAemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTA"]["rhoLTDR"]
            handleAxes = np.empty((sys_info['K'], 2 * sys_info['K']), dtype=object)
        elif plot_info['phi_type'] == 'xi':
            rhoLTRemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTXi"]["rhoLTR"]
            rhoLTAemp[idx] = learningOutput[idx]["rhoLTemp"]["rhoLTXi"]["mrhoLTXi"]
            handleAxes = np.empty((sys_info['K'], 2 * sys_info['K']), dtype=object)

    plot_info["phi_resolution"] = 1001
    for k1 in range(1, sys_info['K']+ 1):
        for k2 in range(1, sys_info['K'] + 1):
            if plot_info['phi_type'] == 'energy':
                axis_handle = plt.subplot(sys_info['K'], sys_info['K'], (k1 - 1) * sys_info['K'] + k2)
            else:
                axis_handle = plt.subplot(sys_info['K'], 2 * sys_info['K'], (k1 - 1) * 2 * sys_info['K'] + 2 * k2 - 1)
            rhoLT = get_single_rhoLT(rhoLTR, k1-1, k2-1)
            for idx in range(total_num_trials):
                rhoLTemp[idx] = get_single_rhoLT(rhoLTRemp[idx], k1-1, k2-1)
                phihat = all_phihat[idx]
                phihats[idx] = phihat[k1 - 1][k2 - 1]
                phihatsmooth = all_phihatsmooth[idx]
                phihatsmooths[idx] = phihatsmooth[(k1 -1, k2 -1)]
            plot_interactions_and_rhos(win_handler, axis_handle, rhoLT, rhoLTemp, phi[(k1-1,k2-1)], phihats, phihatsmooths, k1-1, k2-1, sys_info, plot_info)
            if k1 == sys_info['K']:
                plt.xlabel('pairwise distance', fontsize=plot_info['axis_font_size'], fontname=plot_info['axis_font_name'])
            if plot_info["showplottitles"]:
                sys_N = sys_info['N']
                obs_TL =obs_info['T_L']
                obs_M = obs_info['M']
                obs_L = obs_info['L']
                bas_n = basis_info['n'][k1-1, k2-1]

                titleHandle = plt.title(f'N = {sys_N}, T = {obs_TL}, M = {obs_M}, L = {obs_L}, n = {bas_n}', fontsize=plot_info['title_font_size)'])
                titleHandle.set_position(titleHandle.get_position() * [1, 0.95, 1])
                titleHandle.set_fontweight('bold')
            if plot_info['phi_type'] != 'energy':
                ax = plt.subplot(sys_info['K'], 2 * sys_info['K'], (k1 - 1) * 2 * sys_info['K'] + 2 * k2)
                ax.yaxis.tick_right()
                rhoLT = get_single_rhoLT(rhoLTA, k1-1, k2-1)
                range_vals = get_range_from_rhos(rhoLTemp)
                [rhoPlotHandles, rhoPlotNames] = plot_rhos(ax, range_vals, rhoLT, rhoLTemp, k1-1, k2-1, sys_info, plot_info)
                plt.legend(rhoPlotHandles, rhoPlotNames, loc='upper right', fontsize=plot_info['legend_font_size'], frameon=True, framealpha=1)
                if k1 == sys_info['K']:
                    if plot_info['phi_type'] == 'alignment':
                        xlabel_handle = plt.xlabel('(pairwise speed)', fontsize=plot_info.axis_font_size)
                    elif plot_info['phi_type'] == 'xi':
                        xlabel_handle = plt.xlabel('$\\xi$ (pairwise $\\xi_{i, i\'}$)', fontsize=plot_info['axis_font_size'])
                    ax.xaxis.set_label_coords(0.5, -0.15)
                    ax.set_position(ax.get_position() * [1, 1, 1, 0.95])
                    ax.yaxis.set_tick_params(width=0.5)
                    ax.xaxis.set_tick_params(width=0.5)
                    ax.yaxis.tick_right()
                    ax.set_xlim(0, 1)
            axis_handle.yaxis.set_major_formatter('{:.2g}'.format)
            axis_handle.tick_params(axis='both', which='major', width=0.5)
            axis_handle.tick_params(axis='both', which='minor', width=0.5)

    plt.tight_layout()
    #plt.show(block=False)
    if sys_info['debug_mode']:
        plt.show()
    else:
        plt.ion()



