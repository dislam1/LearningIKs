def get_legend_name_for_rhos(sys_info, plot_info, type, k1, k2):
    if sys_info['K'] > 1:
        type_indicator = f', {k1}{k2}'
    else:
        type_indicator = ''
    
    if type == 'rhoLT':
        if plot_info['phi_type'] == 'energy':
            if sys_info['ode_order'] == 1:
                legend_name = rf'$\rho_T^{{L{type_indicator}}}$'
            else:
                legend_name = rf'$\rho_{{T, r}}^{{L{type_indicator}}}$'
        elif plot_info['phi_type'] == 'alignment':
            legend_name = rf'$\rho_{{T, \dot{{r}}}}^{{L{type_indicator}}}$'
        elif plot_info['phi_type'] == 'xi':
            legend_name = rf'$\rho_{{T, \xi}}^{{L{type_indicator}}}$'
    elif type == 'rhoLTemp':
        if plot_info['phi_type'] == 'energy':
            if sys_info['ode_order'] == 1:
                legend_name = rf'$\rho_T^{{L, M{type_indicator}}}$'
            else:
                legend_name = rf'$\rho_{{T, r}}^{{L, M{type_indicator}}}$'
        elif plot_info['phi_type'] == 'alignment':
            legend_name = rf'$\rho_{{T, \dot{{r}}}}^{{L, M{type_indicator}}}$'
        elif plot_info['phi_type'] == 'xi':
            legend_name = rf'$\rho_{{T, \xi}}^{{L, M{type_indicator}}}$'
    else:
        legend_name = ''

    return legend_name
