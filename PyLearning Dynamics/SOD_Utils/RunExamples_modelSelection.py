import os
import datetime
import numpy as np
from scipy.io import savemat
from utils import LoadExampleDefinitions, SelectExample, generateObservations, learn_from_dynamics, estimateRhoLT, L2_rho_T_energy, L2_rho_T_alignment
import matplotlib.pyplot as plt

# Set parameters
if os.name == 'nt':
    SAVE_DIR = os.path.join(os.getenv('USERPROFILE'), 'DataAnalyses', 'LearningDynamics')
else:
    SAVE_DIR = os.path.join(os.getenv('HOME'), 'DataAnalyses', 'LearningDynamics')

VERBOSE = 1  # Indicator to print certain output
if 'Params' not in locals():
    Params = []
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load example definitions and let the user select one example to run
Examples = LoadExampleDefinitions('ModelSelection')
ExampleIdx = SelectExample(Params, Examples)

# Get example parameters
Example = Examples[ExampleIdx]
sys_info = Example['sys_info']
solver_info = Example['solver_info']
obs_info = Example['obs_info']
learn_info = Example['learn_info']
plot_info = Example['plot_info']
learn_info['is_parallel'] = False  # Some fine-tuning of learning parameters
learn_info['keep_obs_data'] = True
learn_info['VERBOSE'] = VERBOSE
learn_info['SAVE_DIR'] = SAVE_DIR
learn_info['MEMORY_LEAN'] = True
obs_info['compute_pICs'] = False
obs_info['VERBOSE'] = VERBOSE
obs_info['SAVE_DIR'] = SAVE_DIR
n_trials = 1
learningOutput = [None] * n_trials

# Start parallel pool (if needed)
# gcp()

time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
learn_tic = datetime.datetime.now()

if ExampleIdx == 1 or ExampleIdx == 2:
    for idx in range(n_trials):
        print(f'\n=========================Trial ID: {idx+1}')
        print(f'Generating {obs_info["M"]} trajectories with {obs_info["L"]} observations each...', end='')
        obs_data = generateObservations(sys_info, solver_info, obs_info, obs_info["M"], [])
        print(f'done ({obs_data["Timings"]["total"]:.2f} sec).')

        print(f'\n=========================Trial ID: {idx+1}')
        print('Learning...', end='')
        learn_time = datetime.datetime.now()

        learn_out = learn_from_dynamics(sys_info, obs_info, learn_info, obs_data)
        learn_out['rhoLTemp'] = estimateRhoLT(obs_data, sys_info, obs_info)

        print(f'done ({(datetime.datetime.now() - learn_time).total_seconds():.2f} sec).')
        print(f'The empirical error from learning phis on x/v is: {learn_out["Estimator"]["emp_err"]:10.4e}.')

        obs_info['rhoLT'] = learn_out['rhoLTemp']

        print(f'\n=========================Trial ID: {idx+1}')
        print('Computing corresponding L^2(rho^L_T) norms of estimators...', end='')

        learn_out['L2rhoTE'] = L2_rho_T_energy(learn_out['Estimator']['phiEhat'], sys_info, obs_info, learn_out['Estimator']['Ebasis'])
        learn_out['L2rhoTA'] = L2_rho_T_alignment(learn_out['Estimator']['phiAhat'], sys_info, obs_info, learn_out['Estimator']['Abasis'])

        print(f'\nThe L^2(rho^L_T) of the learned estimator for phi^E is: {learn_out["L2rhoTE"][0, 0]:12.6e}.')
        print(f'The L^2(rho^L_T) of the learned estimator for phi^A is: {learn_out["L2rhoTA"][0, 0]:12.6e}.')

        learningOutput[idx] = learn_out

    print(f'\n=========================After {n_trials} trials')
    L2rhoTE_vals = np.array([out['L2rhoTE'][0, 0] for out in learningOutput])
    L2rhoTA_vals = np.array([out['L2rhoTA'][0, 0] for out in learningOutput])

    print(f'The L^2(rho^L_T) of the learned estimator for phi^E is: {np.mean(L2rhoTE_vals):12.6e} ± {np.std(L2rhoTE_vals):12.6e}.')
    print(f'The L^2(rho^L_T) of the learned estimator for phi^A is: {np.mean(L2rhoTA_vals):12.6e} ± {np.std(L2rhoTA_vals):12.6e}.')

    plot_info['plot_name'] = os.path.join(SAVE_DIR, f"{sys_info['name']}_learningOutput_{time_stamp}")

    # Display and save results
    # displayMSLearningResults(learningOutput, sys_info, plot_info)  # Implement this function for displaying results

    # Save
    save_dict = {
        'sys_info': sys_info,
        'solver_info': solver_info,
        'obs_info': obs_info,
        'learn_info': learn_info,
        'obs_data': obs_data,
        'learningOutput': learningOutput,
        'plot_info': plot_info
    }

    savemat(os.path.join(SAVE_DIR, f"{sys_info['name']}_learningOutput{time_stamp}.mat"), save_dict, appendmat=False, format='5')

elif ExampleIdx == 3 or ExampleIdx == 4:
    # You can continue with the translation of the rest of the code for ExampleIdx 3 and 4
    pass

print(f'\nOverall time for conducting Model Selection: {(datetime.datetime.now() - learn_tic).total_seconds():.2f}')

# Done
print('\ndone.\n')
