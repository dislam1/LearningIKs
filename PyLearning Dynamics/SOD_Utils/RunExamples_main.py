import os
from pytictoc import TicToc
#import SOD_Utils.Load_Example_Definitions, SOD_Utils.Select_Example, SOD_Utils.restructure_sys_info_for_larger_N, SOD_Utils.computeL2rhoTErr, SOD_Utils.final_visualization
from SOD_Utils.Load_Example_Definitions import LoadExampleDefinitions
from SOD_Utils.Select_Example import SelectExample

from SOD_Learn.generateRhoLT import generateRhoLT
from SOD_Learn.learningRoutine import learningRoutine
from SOD_Learn.estimateTrajAccuracies import estimateTrajAccuracies
from SOD_Utils.computeL2rhoTErr import computeL2rhoTErr
from SOD_Utils.restructure_sys_info_for_larger_N import restructure_sys_info_for_larger_N
from SOD_Utils.final_visualization import final_visualization

# Set parameters
if os.name == 'nt':
    SAVE_DIR = os.path.join(os.environ['USERPROFILE'], 'DataAnalyses', 'LearningDynamics')
else:
    SAVE_DIR = os.path.join(os.environ['HOME'], 'DataAnalyses', 'LearningDynamics')

VERBOSE = 1
Params = None
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load example definitions and let user select one example to run

Examples = LoadExampleDefinitions()
ExampleIdx = SelectExample(Params, Examples)

# Get example parameters
Example = Examples[ExampleIdx]
#print(Example)
sys_info = Example['sys_info']
solver_info = Example['solver_info']
obs_info = Example['obs_info']
learn_info = Example['learn_info']
plot_info = Example['plot_info']
#print(solver_info)
# Other settings
reuse_rho_T = False
n_trials = 1

learn_info['to_predict_LN'] = True
learn_info['is_parallel'] = False
learn_info['keep_obs_data'] = True
learn_info['VERBOSE'] = VERBOSE
learn_info['SAVE_DIR'] = SAVE_DIR
learn_info['MEMORY_LEAN'] = True
obs_info['compute_pICs'] = False
obs_info['VERBOSE'] = VERBOSE
obs_info['SAVE_DIR'] = SAVE_DIR

# obs_info['obs_noise'] = 0.0
if obs_info['obs_noise'] > 0:
    obs_info['use_derivative'] = True

# Start parallel pool
# gcp()  # Not required in Python

time_stamp = 'your_date_string'  # Replace with a valid date string

# Generate rho^L_T if needed
print('\n================================================================================')
print('\nGenerating rhoT......')
tictoc = TicToc()
tictoc.tic()
obs_info['rhoLT'] = generateRhoLT(sys_info, solver_info, obs_info, reuse_rho_T)
tictoc.toc()
print(f'done ({tictoc.elapsed})')
#exit
# Perform learning and test performance on trajectories
learningOutput = []
learn_info['sys_info'] = sys_info.copy()
print('\n================================================================================')
print('\nLearning Interaction Law(s)......')
for trial_idx in range(n_trials):
    if VERBOSE >= 1:
        print(f'\n------------------- Learning with trial ID#{trial_idx + 1}.')
    print(learn_info)
    learn_out = learningRoutine(solver_info, obs_info, learn_info)
    learningOutput.append(learn_out)

# Test the performance on phi_hats (and regularized phi_hats)
print('\n================================================================================')
tictoc.tic()
print('\nComputing L2(rhoT) Errors......')
rStr = ''
for k in range(n_trials - 1, -1, -1):
    learningOutput[k]['L2rhoTErr'] = computeL2rhoTErr(learningOutput[k]['Estimator'], sys_info, obs_info)
    msg = f'{k + 1:3d}'
    print(rStr + msg, end='')
    rStr = '\b' * len(msg)
tictoc.toc()
print(f'\nOverall time for computing L2(rho_T) errors: {tictoc.elapsed}')

# Test performance on trajectories
print('\n================================================================================')
tictoc.tic()
print('\nComputing Trajectory Errors......')
rStr = ''
#Keep copy of sys_info since we are going to modify.
sys_info_cp = sys_info.copy()


if 'to_predict_LN' in learn_info and learn_info['to_predict_LN']:
    obs_info['N_ratio'] = learn_info['N_ratio']
    sys_info_Ntransfer = restructure_sys_info_for_larger_N(learn_info['N_ratio'], sys_info_cp)
else:
    sys_info_Ntransfer = None

print('\n------------------- Trajectory Error with trial ID#: ')


for k in range(n_trials - 1, -1, -1):
    learningOutput[k]['Timings']['estimateTrajAccuracies'] = TicToc()
    learningOutput[k]['Timings']['estimateTrajAccuracies'].tic()
    (
        learningOutput[k]['trajErr'],
        learningOutput[k]['trajErr_new'],
        learningOutput[k]['y_init_new'],
        learningOutput[k]['trajErr_Ntransfer'],
        learningOutput[k]['y_init_Ntransfer'],
        learningOutput[k]['syshatsmooth_info_Ntransfer']
    ) = estimateTrajAccuracies(sys_info, learningOutput[k]['syshatsmooth_info'], learningOutput[k]['obs_data'], obs_info, solver_info, sys_info_Ntransfer)
    learningOutput[k]['Timings']['estimateTrajAccuracies'].toc()
    msg = f'{k + 1:3d}'
    print(rStr + msg, end='')
    rStr = '\b' * len(msg)
tictoc.toc()
print(f'\nOverall time for computing trajectory errors: {tictoc.elapsed}')

# Save
#plot_info['save_file'] = os.path.join(SAVE_DIR, f'{sys_info["name"]}_learningOutput_{time_stamp}.mat')
import scipy.io as sio

"""
sio.savemat(
    plot_info['save_file'],
    {
        'sys_info': sys_info,
        'solver_info': solver_info,
        'obs_info': obs_info,
        'learn_info': learn_info,
        'plot_info': plot_info,
        'learningOutput': learningOutput,
        'sys_info_Ntransfer': sys_info_Ntransfer,
        'time_stamp': time_stamp,
    },
    appendmat=False,
    format='5',
)
"""

plot_info_save = {
    'sys_info': sys_info,
    'solver_info': solver_info,
    'obs_info': obs_info,
    'learn_info': learn_info,
    'plot_info': plot_info,
    'learningOutput': learningOutput,
    'sys_info_Ntransfer': sys_info_Ntransfer,
    'time_stamp': time_stamp,
    'appendmat': False,
    'format' : 5
}

with open('C:/Users/CluClu/Downloads/Maggioni Python/SOD_Utils/ode_data.txt','w') as data: 
      data.write(str(plot_info_save))
   


# Display & figures
if VERBOSE >= 1:
    final_visualization(learningOutput, obs_info, solver_info, sys_info, sys_info_Ntransfer, learn_info, time_stamp, plot_info)

# Done
print('\ndone.')
#if __name__ == "__main__":
