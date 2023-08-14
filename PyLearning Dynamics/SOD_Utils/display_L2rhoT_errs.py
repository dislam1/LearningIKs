def displayL2rhoTerrs(learning_output, sys_info):
    # \phi^E
    if sys_info['phiE']:
        Err = []
        Err_smooth = []
        for trial_idx in range(len(learning_output)):
            Err.append(learning_output[trial_idx]['L2rhoTErr']['EErr'])
            Err_smooth.append(learning_output[trial_idx]['L2rhoTErr']['EErrSmooth'])
        print_one_L2rhoT_err(sys_info['K'], Err, Err_smooth, 'energy')

    # \phi^A
    if sys_info['ode_order'] == 2 and sys_info['phiA']:
        Err = []
        Err_smooth = []
        for trial_idx in range(len(learning_output)):
            Err.append(learning_output[trial_idx]['L2rhoTErr']['AErr'])
            Err_smooth.append(learning_output[trial_idx]['L2rhoTErr']['AErrSmooth'])
        print_one_L2rhoT_err(sys_info['K'], Err, Err_smooth, 'alignment')

    # \phi^\xi
    if sys_info['ode_order'] == 2 and sys_info['has_xi']:
        Err = []
        Err_smooth = []
        for trial_idx in range(len(learning_output)):
            Err.append(learning_output[trial_idx]['L2rhoTErr']['XiErr'])
            Err_smooth.append(learning_output[trial_idx]['L2rhoTErr']['XiErrSmooth'])
        print_one_L2rhoT_err(sys_info['K'], Err, Err_smooth, 'xi')


def print_one_L2rhoT_err(K, Err, Err_smooth, component):
    # Implement this function according to your needs.
    # It is responsible for displaying or printing the L2-norm error estimates.
    # The arguments `K`, `Err`, and `Err_smooth` correspond to the number of components, 
    # error estimates, and smoothed error estimates for the specified `component`.
    # Depending on your application, you can visualize or log the error data.
    # The exact implementation will depend on how the errors are calculated and what kind of output you want.
    # You might use plotting libraries like Matplotlib in Python to create visualizations.
    # Feel free to customize this function to suit your specific requirements.
    pass
