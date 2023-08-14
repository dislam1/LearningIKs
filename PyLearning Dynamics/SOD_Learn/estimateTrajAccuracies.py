from SOD_Utils.computeTrajErr import computeTrajErr
from SOD_Utils.restructure_sys_info_for_larger_N import restructure_sys_info_for_larger_N
from SOD_Utils.generateICs import generateICs

def estimateTrajAccuracies(sys_info, syshat_info, obs_data, obs_info, solver_info, sys_info_Ntransfer=None):
    # function [trajErr, trajErr_new, y_init_new, trajErr_Ntransfer, y_init_Ntransfer, syshat_info_Ntransfer] = ...
    #      estimateTrajAccuracies(sys_info, syshat_info, obs_data, obs_info, solver_info, sys_info_Ntransfer)
    #
    # (c) M. Zhong (JHU)

    # do an input parameter list check
    
    if sys_info_Ntransfer is None:
        sys_info_Ntransfer = {}

    trajErr = computeTrajErr(sys_info, syshat_info, solver_info, obs_info, obs_data['ICs'])

    # Measure the performance in terms of errors on trajectories from the same initial conditions as training
   # trajErr = computeTrajErr(sys_info, syshat_info, solver_info, obs_info, obs_data['ICs'])


    # Measure the performance in terms of errors on trajectories from new initial conditions
    y_init_new = generateICs(obs_info['M'], sys_info)
    trajErr_new = computeTrajErr(sys_info, syshat_info, solver_info, obs_info, y_init_new)

    # Measure the performance in terms of errors on trajectories from systems with larger N
    syshat_info_Ntransfer = {}
    y_init_Ntransfer = []
    trajErr_Ntransfer = []

    if sys_info_Ntransfer:
        sys_info1 = sys_info.copy()
        syshat_info_Ntransfer = restructure_sys_info_for_larger_N(obs_info['N_ratio'], syshat_info)

        y_init_Ntransfer = generateICs(obs_info['M'], sys_info_Ntransfer)
        trajErr_Ntransfer = computeTrajErr(sys_info_Ntransfer, syshat_info_Ntransfer, solver_info, obs_info, y_init_Ntransfer)
        sys_info = sys_info1.copy()

    return trajErr, trajErr_new, y_init_new, trajErr_Ntransfer, y_init_Ntransfer, syshat_info_Ntransfer

'''
Please note that the code assumes that the functions computeTrajErr(), generateICs(), 
and restructure_sys_info_for_larger_N() are defined elsewhere in your code. 
You'll need to provide their definitions for the code to work properly.
'''