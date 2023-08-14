def get_legend_name_for_phis(sys_info, plot_info, type, k1, k2):
    if sys_info['K'] > 1:
        type_indicator = '_{' + str(k1) + str(k2) + '}'
    else:
        type_indicator = ''

    if type == 'phi':
        if plot_info['phi_type'] == 'energy':
            if sys_info['ode_order'] == 1:
                legend_name = r'$\phi' + type_indicator + '$'
            else:
                legend_name = r'$\phi^E' + type_indicator + '$'
        elif plot_info['phi_type'] == 'alignment':
            legend_name = r'$\phi^A' + type_indicator + '$'
        elif plot_info['phi_type'] == 'xi':
            legend_name = r'$\phi^{\xi}' + type_indicator + '$'
    elif type == 'phihat':
        if plot_info['phi_type'] == 'energy':
            if sys_info['ode_order'] == 1:
                legend_name = r'$\hat\phi' + type_indicator + '$'
            else:
                legend_name = r'$\hat\phi^E' + type_indicator + '$'
        elif plot_info['phi_type'] == 'alignment':
            legend_name = r'$\hat\phi^A' + type_indicator + '$'
        elif plot_info['phi_type'] == 'xi':
            legend_name = r'$\hat\phi^{\xi}' + type_indicator + '$'
    elif type == 'phihatsmooth':
        if plot_info['phi_type'] == 'energy':
            if sys_info['ode_order'] == 1:
                legend_name = r'$\phi^{\text{reg}}' + type_indicator + '$'
            else:
                legend_name = r'$\phi^{E, \text{reg}}' + type_indicator + '$'
        elif plot_info['phi_type'] == 'alignment':
            legend_name = r'$\phi^{A, \text{reg}}' + type_indicator + '$'
        elif plot_info['phi_type'] == 'xi':
            legend_name = r'$\phi^{\xi, \text{reg}}' + type_indicator + '$'
    else:
        legend_name = ''

    return legend_name
