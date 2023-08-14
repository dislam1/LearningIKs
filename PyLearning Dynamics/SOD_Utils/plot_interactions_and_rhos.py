import matplotlib.pyplot as plt
import numpy as np
from SOD_Utils.get_range_from_rhos import get_range_from_rhos
from SOD_Utils.plot_rhos import plot_rhos
from SOD_Utils.get_exponent_scale import get_exponent_scale
from SOD_Utils.get_legend_name_for_phis import get_legend_name_for_phis
from SOD_Utils.eval_basis_functions import eval_basis_functions
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_prey import PS_1st_order_prey_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_prey import PS_1st_order_predator_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_predator import PS_1st_order_prey_on_predator
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_predator import PS_1st_order_predator_on_predator

def plot_interactions_and_rhos(fig_handle, axis_handle, rhoLT, rhoLTemp, phi, phihats, phihatsmooths, k1, k2, sys_info, plot_info):
    fig = plt.figure(fig_handle)
    #For testing
    #plt.sca(axis_handle)
    plt.sca(axis_handle)



    total_num_trials = len(rhoLTemp)
    range_val = get_range_from_rhos(rhoLTemp)
    if abs(range_val[1] - range_val[0]) < 1.0e-12:
        range_val[1] = range_val[0] + 1
    r = np.linspace(range_val[0], range_val[1], plot_info['phi_resolution'])
    pts = r
    if phi is not None:
        phir = eval(phi)(pts)
    else:
        phir = np.empty(0)
    phihatr = np.zeros((total_num_trials, len(r)))
    phihatsmoothr = np.zeros((total_num_trials, len(r)))
    

    #Change plot_info dictionary
    #plot_info = plot_info['plot_info']

    for ind in range(total_num_trials):
        phihat = phihats[ind]
        #Construct the lambda function
        alpha_vec = phihat['alpha_vec']
        basis = phihat['basis']    
        fun = phihat['fun']
        fval, fprime = (lambda r : eval_basis_functions(r, alpha_vec, basis) )(pts)
        phihatr[ind, :] = fval
    

        if plot_info['display_interpolant']:
            phihatsmooth = phihatsmooths[ind]
            phihatsmoothr[ind, :] = phihatsmooth(r)
        else:
            phihatsmoothr[ind, :] = phihatr[ind, :]

    y_min = max([np.min(phir[phir > -np.Inf]), np.min(phihatr), np.min(phihatsmoothr[phihatsmoothr > -np.Inf])])
    y_max = min([np.max(phir[phir < np.Inf]), np.max(phihatr[phihatr < np.Inf]), np.max(phihatsmoothr[phihatsmoothr < np.Inf])])
    
    if y_max < y_min + 10 * np.finfo(float).eps:
        y_max = y_min + 1
        y_min = y_min - 1
    
    #plt.gca().yaxis.set_label_coords(1.02, 0.5)
    plt.gca().yaxis.tick_right()
    
    # Display \rho^L_T and its estimator
    ax = plt.gca()
    rhoPlotHandles, rhoPlotNames = plot_rhos(fig_handle, ax, range_val, rhoLT, rhoLTemp, k1, k2, sys_info, plot_info)
    plt.gca().yaxis.tick_left()
    # Display interaction kernels
    ax = plt.gca()
    font = {'family' : plot_info['tick_font_name'],
        'weight' : 'bold',
        'size'   : plot_info['tick_font_size']}
    plt.rc('font',**font)

    ax.set_xlabel('$r$ (pairwise distance)', fontsize=plot_info['axis_font_size'])
    ax.set_ylabel('$r$ (Interaction Kernels)', fontsize=plot_info['axis_font_size'])
    ax.tick_params(axis='both', labelsize=plot_info['tick_font_size'], labelcolor='black')
    ax.yaxis.label.set_color('black')
    
    num_PPH = 3 if plot_info['display_interpolant'] and plot_info['display_phihat'] else 2
    if not phir.size:
        num_PPH -= 1
    
    phiPlotHandles = []
    phiPlotNames = []
    PPH_count = 0
    
    if phir.size > 0:
        PPH_count += 1
        phiPlotHandles.append(plt.plot(r, phir, 'k', linewidth=plot_info['phi_line_width'])[0])
        phiPlotNames.append(get_legend_name_for_phis(sys_info, plot_info, 'phi', k1, k2))
        plt.plot(r, np.zeros_like(r), 'k--')
    else:
        plt.plot(r, np.zeros_like(r), 'k--')
    #Commenting all the mean display
        
    if plot_info['display_phihat'] or not plot_info['display_interpolant']:
        PPH_count += 1
        mean_phihatr = np.mean(phihatr, axis=0)
        phiPlotHandles.append(plt.plot(r, mean_phihatr, '-r', linewidth=plot_info['phihat_line_width'])[0])
        phiPlotNames.append(get_legend_name_for_phis(sys_info, plot_info, 'phihat', k1, k2))
        plt.plot(r, mean_phihatr + np.std(phihatr, axis=0), '--r', linewidth=plot_info['phihat_line_width'] / 4)
        plt.plot(r, mean_phihatr - np.std(phihatr, axis=0), '--r', linewidth=plot_info['phihat_line_width'] / 4)
    
    if plot_info['display_interpolant']:
        PPH_count += 1
        mean_phihatsmoothr = np.mean(phihatsmoothr, axis=0)
        phiPlotHandles.append(plt.plot(r, mean_phihatsmoothr, '-b', linewidth=plot_info['phihat_line_width'])[0])
        if not plot_info['display_phihat']:
            phiPlotNames.append(get_legend_name_for_phis(sys_info, plot_info, 'phihat', k1, k2))
        else:
            phiPlotNames.append(get_legend_name_for_phis(sys_info, plot_info, 'phihatsmooth', k1, k2))
        plt.plot(r, mean_phihatsmoothr + np.std(phihatsmoothr, axis=0), '--b', linewidth=plot_info['phihat_line_width'] / 4)
        plt.plot(r, mean_phihatsmoothr - np.std(phihatsmoothr, axis=0), '--b', linewidth=plot_info['phihat_line_width'] / 4)
    
    legendHandle = ax.legend(phiPlotHandles + rhoPlotHandles, phiPlotNames + rhoPlotNames, loc='upper right', 
                             fontsize=plot_info['legend_font_size'], prop={'family': plot_info['legend_font_name']})
    
    y_range = y_max - y_min
    y_min = y_min 
    y_max = y_max + y_range * 0.5
    plt.axis([range_val[0], range_val[1], y_min, y_max])
    
    the_max = max(abs(y_min), abs(y_max))
    #exp_n = get_exponent_scale(the_max)
    #ax.ticklabel_format(style='sci', scilimits=(exp_n, exp_n), axis='y')
    #ax.yaxis.set_major_formatter('{:+.2g}'.format)
    #plt.show(block = False)
    #from matplotlib import ticker
    #formatter = ticker.ScalarFormatter(useMathText=True)
    #formatter.set_scientific(True) 
    #formatter.set_powerlimits((y_min,y_max)) 
    #ax.yaxis.set_major_formatter(formatter) 

    if sys_info['debug_mode']:
        plt.show()
    else:
        plt.ion()
