import numpy as np
import matplotlib.pyplot as plt
import os
from SOD_Utils.displayTrajErrs import displayTrajerrs
from SOD_Utils.displayLearningResults import  displaylearningresults
from SOD_Utils.visualize_trajs import visualizetrajs
from SOD_Utils.display_L2rhoT_errs import displayL2rhoTerrs

def final_visualization(learning_output, obs_info, solver_info, sys_info, sys_info_Ntransfer, learn_info, time_stamp, plot_info):
    # displaying errors
    print('\n================================================================================')
    print(f'\n------------------- Errors over {len(learning_output)} Learning Trial(s)')
    if 'trajErr' in learning_output[0]:
        displayTrajerrs(learning_output, obs_info, solver_info)  # Implement displayTrajErrs function in Python
    if 'L2rhoTErr' in learning_output[0]:
        displayL2rhoTerrs(learning_output, sys_info)  # Implement displayL2rhoTErrs function in Python

    # displaying results
    print('\n------------------- Visualizing phis and trajs')
    plot_info['plot_name'] = os.path.join(learn_info['SAVE_DIR'], f'{sys_info["name"]}_learningOutput_{time_stamp}')
    plot_info['solver_info'] = solver_info
    plot_info['for_larger_N'] = False
    displaylearningresults(learning_output, sys_info, learning_output[0]['syshatsmooth_info'], obs_info, learn_info, learning_output[0]['obs_data']['ICs'], plot_info)  # Implement displayLearningResults function in Python

    # for Larger N
    if sys_info_Ntransfer is not None:
        plot_info['for_larger_N'] = True
        sys_info1 = sys_info.copy()
        visualizetrajs(learning_output, sys_info_Ntransfer, learning_output[0]['syshatsmooth_info_Ntransfer'], obs_info, learning_output[0]['y_init_Ntransfer'], plot_info)  # Implement visualize_trajs function in Python
        print("final_visualization - method visualizetrajs")
        
    plt.show(block=True)

