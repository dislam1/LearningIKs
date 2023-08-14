import numpy as np

def displayTrajerrs(learning_output, obs_info, solver_info):
    # Using the same initial conditions as the training data
    sup_norm_same_ic = [np.mean([trial['trajErr']['sup'] for trial in learning_output]),
                        np.std([trial['trajErr']['sup'] for trial in learning_output])]
    sup_norm_fut_same_ic = [np.mean([trial['trajErr']['sup_fut'] for trial in learning_output]),
                            np.std([trial['trajErr']['sup_fut'] for trial in learning_output])]

    # Using new initial conditions
    sup_norm_new_ic = [np.mean([trial['trajErr_new']['sup'] for trial in learning_output]),
                       np.std([trial['trajErr_new']['sup'] for trial in learning_output])]
    sup_norm_fut_new_ic = [np.mean([trial['trajErr_new']['sup_fut'] for trial in learning_output]),
                           np.std([trial['trajErr_new']['sup_fut'] for trial in learning_output])]

    # Using larger N (number of samples)
    if hasattr(learning_output[0], 'trajErr_Ntransfer') and learning_output[0].trajErr_Ntransfer:
        sup_norm_larger_N = [np.mean([trial['trajErr_Ntransfer']['sup'] for trial in learning_output]),
                             np.std([trial['trajErr_Ntransfer']['sup'] for trial in learning_output])]
        sup_norm_fut_larger_N = [np.mean([trial['trajErr_Ntransfer']['sup_fut'] for trial in learning_output]),
                                 np.std([trial['trajErr_Ntransfer']['sup_fut'] for trial in learning_output])]
        # Display the results
        print(f"\n------------------- Trajectory accuracies, larger N:"
              f"\n\tsup-norm on [0, {obs_info.T_L}], mean = {sup_norm_larger_N[0]:.4e}±{sup_norm_larger_N[1]:.4e}, "
              f"std = {sup_norm_larger_N[2]:.4e}±{sup_norm_larger_N[3]:.4e}"
              f"\n\tsup-norm on [{obs_info.T_L}, {solver_info.time_span[1]}], "
              f"mean = {sup_norm_fut_larger_N[0]:.4e}±{sup_norm_fut_larger_N[1]:.4e}, "
              f"std = {sup_norm_fut_larger_N[2]:.4e}±{sup_norm_fut_larger_N[3]:.4e}")
    else:
        sup_norm_larger_N = [0, 0]
        sup_norm_fut_larger_N = [0, 0]

    # Display the results
    print(f"\n------------------- Trajectory accuracies, same IC's as training data:"
          f"\n\tsup-norm on [0, {obs_info['T_L']}], mean = {sup_norm_same_ic[0]:.4e}±{sup_norm_same_ic[1]:.4e}, "
          f"std = {sup_norm_same_ic[0]:.4e}±{sup_norm_same_ic[1]:.4e}"
          f"\n\tsup-norm on [{obs_info['T_L']}, {solver_info['time_span'][1]}], "
          f"mean = {sup_norm_fut_same_ic[0]:.4e}±{sup_norm_fut_same_ic[1]:.4e}, "
          f"std = {sup_norm_fut_same_ic[0]:.4e}±{sup_norm_fut_same_ic[1]:.4e}")

    print(f"\n------------------- Trajectory accuracies, new IC's:"
          f"\n\tsup-norm on [0, {obs_info['T_L']}], mean = {sup_norm_new_ic[0]:.4e}±{sup_norm_new_ic[1]:.4e}, "
          f"std = {sup_norm_new_ic[0]:.4e}±{sup_norm_new_ic[1]:.4e}"
          f"\n\tsup-norm on [{obs_info['T_L']}, {solver_info['time_span'][1]}], "
          f"mean = {sup_norm_fut_new_ic[0]:.4e}±{sup_norm_fut_new_ic[1]:.4e}, "
          f"std = {sup_norm_fut_new_ic[0]:.4e}±{sup_norm_fut_new_ic[1]:.4e}")
