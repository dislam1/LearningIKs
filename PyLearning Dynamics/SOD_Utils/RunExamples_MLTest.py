import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from utils import LoadExampleDefinitions, SelectExample, generateRhoLT, learningRoutine, computeL2rhoTErr, generateObservations

# Set parameters
global SAVE_DIR, VERBOSE
if os.name == 'nt':
    SAVE_DIR = os.path.join(os.environ['USERPROFILE'], 'DataAnalyses', 'LearningDynamics')
else:
    SAVE_DIR = os.path.join(os.environ['HOME'], 'DataAnalyses', 'LearningDynamics')
VERBOSE = 0

TEST_ON_TRAJECTORIES = False

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load example definitions and let the user select one example to run
Examples = LoadExampleDefinitions()
ExampleIdx = SelectExample(None, Examples)

# Get example parameters
Example = Examples[ExampleIdx]
sys_info = Example['sys_info']
solver_info = Example['solver_info']
obs_info = Example['obs_info']
learn_info = Example['learn_info']

reuse_rho_T = True
reuse_trajectories = True
reuse_runs = True
save_trajectories = True
save_results_at_every_run = False
n_trials = 1

# Parameters specific to experiments for the M-L plane plots
n_MAX = 1024
M_plane = np.unique([1, max(1, np.ceil(2.0 ** np.arange(-4, 5) * obs_info['M']))])
L_plane = np.unique(np.maximum(1, np.ceil(2.0 ** np.arange(-4, 3) * obs_info['L'])))
n_plane = [lambda x, y: 2 * np.minimum(x, n_MAX), lambda x, y: 2 * np.ceil(np.sqrt(np.minimum(x, n_MAX))),
           lambda x, y: 2 * np.ceil(np.minimum(y, n_MAX)), lambda x, y: 2 * np.minimum(x * y, n_MAX),
           lambda x, y: 2 * np.ceil(np.sqrt(np.minimum(x * y, n_MAX)))]

obs_info['use_derivative'] = True

learn_info['sys_info'] = sys_info
learn_info['is_parallel'] = False
learn_info['keep_obs_data'] = True
learn_info['VERBOSE'] = VERBOSE
learn_info['SAVE_DIR'] = SAVE_DIR
learn_info['MEMORY_LEAN'] = True
obs_info['compute_pICs'] = False
obs_info['VERBOSE'] = VERBOSE
obs_info['SAVE_DIR'] = SAVE_DIR

# Generate \rho^L_T if needed
print('\n================================================================================')
print('\nGenerating rhoT...')
tic_rhoLT = time.time()
obs_info['rhoLT'] = generateRhoLT(sys_info, solver_info, obs_info, reuse_rho_T)
print(f'done ({time.time() - tic_rhoLT:.2f}s).')

# Perform Learning for all values of M and L
obs_info_time_I = [obs_info['time_vec'][0], obs_info['time_vec'][-1]]  # Use fixed time interval

if 'learningOutput' not in locals():
    learningOutput = [[[] for _ in range(len(n_plane))] for _ in range(len(L_plane))]

print('\nGenerating trajectories...')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for k in range(1, n_trials + 1):
    filename = os.path.join(SAVE_DIR, f'{sys_info["name"]}_traj_{k}.mat')
    if not reuse_trajectories or not os.path.exists(filename):
        Timings_generateObservations = time.time()
        # Use the same trajectories for all
        print(f'\nGenerating observations for trial {k}/{n_trials}...', end='')
        tic_obs = time.time()
        obs_data_all = generateObservations(sys_info, solver_info, obs_info, int(max(M_plane)))
        obs_data_all['x'] = []
        obs_data_all['xp'] = []
        print(f'done ({time.time() - tic_obs:.2f}s).')
        Timings_generateObservations = time.time() - Timings_generateObservations

        if save_trajectories:
            savemat(filename, {
                'sys_info': sys_info,
                'solver_info': solver_info,
                'obs_info': obs_info,
                'obs_data_all': obs_data_all,
                'Timings': {'generateObservations': Timings_generateObservations}
            }, appendmat=False, format='5')
    else:
        print('loading...', end='')
        data = loadmat(filename)
        obs_data_all = data['obs_data_all']
        Timings_generateObservations = data['Timings'][0, 0]['generateObservations'][0, 0]
        print('done.')

    Timings_total = np.zeros(len(M_plane), dtype='uint64')
    for Midx in range(len(M_plane)):
        obs_info['M'] = M_plane[Midx]
        Timings_total[Midx] = time.time()
        for Lidx in range(len(L_plane)):
            obs_info_cur = obs_info.copy()
            obs_info_cur['L'] = L_plane[Lidx]
            if obs_info_cur['M'] * obs_info_cur['L'] < 2:
                continue
            obs_info_cur['time_vec'] = np.linspace(obs_info_time_I[0], obs_info_time_I[1], obs_info_cur['L'])

            nidx = 0
            learn_info_cur = learn_info.copy()
            if 'Ebasis_info' in learn_info_cur and learn_info_cur['Ebasis_info'] is not None:
                learn_info_cur['Ebasis_info']['n'] = n_plane[nidx](obs_info_cur['M'], obs_info_cur['L']) * np.ones((sys_info['K'], sys_info['K']))
                ntmp = learn_info_cur['Ebasis_info']['n']
            if 'Abasis_info' in learn_info_cur and learn_info_cur['Abasis_info'] is not None:
                learn_info_cur['Abasis_info']['n'] = n_plane[nidx](obs_info_cur['M'], obs_info_cur['L']) * np.ones((sys_info['K'], sys_info['K']))
                ntmp = learn_info_cur['Abasis_info']['n']
            if 'Xibasis_info' in learn_info_cur and learn_info_cur['Xibasis_info'] is not None:
                learn_info_cur['Xibasis_info']['n'] = n_plane[nidx](obs_info_cur['M'], obs_info_cur['L']) * np.ones((sys_info['K'], sys_info['K']))
                ntmp = learn_info_cur['Xibasis_info']['n']

            tic_learningroutine = time.time()
            print(f'\nM({Midx + 1}/{len(M_plane)})={obs_info_cur["M"]}, L({Lidx + 1}/{len(L_plane)})={obs_info_cur["L"]}, n({nidx + 1}/{len(n_plane)})={ntmp}...', end='')
            if reuse_runs and len(learningOutput[Lidx][Midx][nidx]) > 0:
                print('...skipping...', end='')
                continue

            learningOutput[Lidx][Midx][nidx].append(learningRoutine(solver_info, obs_info_cur, learn_info_cur, obs_data_all))  # Learning
            learningOutput[Lidx][Midx][nidx][-1]['L2rhoTErr'] = computeL2rhoTErr(learningOutput[Lidx][Midx][nidx][-1]['Estimator'], sys_info, obs_info_cur)
            print(f'done ({time.time() - tic_learningroutine:.2f}s).')

            for nidx in range(1, len(n_plane)):
                tic_learningroutine = time.time()
                n_actual = n_plane[nidx](obs_info_cur['M'], obs_info_cur['L']) * np.ones((sys_info['K'], sys_info['K']))
                learn_info_cur = learn_info.copy()
                if 'Ebasis_info' in learn_info_cur and learn_info_cur['Ebasis_info'] is not None:
                    learn_info_cur['Ebasis_info']['n'] = n_actual
                    ntmp = learn_info_cur['Ebasis_info']['n']
                if 'Abasis_info' in learn_info_cur and learn_info_cur['Abasis_info'] is not None:
                    learn_info_cur['Abasis_info']['n'] = n_actual
                    ntmp = learn_info_cur['Abasis_info']['n']
                if 'Xibasis_info' in learn_info_cur and learn_info_cur['Xibasis_info'] is not None:
                    learn_info_cur['Xibasis_info']['n'] = n_actual
                    ntmp = learn_info_cur['Xibasis_info']['n']

                print(f'\nM({Midx + 1}/{len(M_plane)})={obs_info_cur["M"]}, L({Lidx + 1}/{len(L_plane)})={obs_info_cur["L"]}, n({nidx + 1}/{len(n_plane)})={ntmp}...', end='')
                learningOutput[Lidx][Midx].append([])
                learningOutput[Lidx][Midx][-1].append(learningRoutine(solver_info, obs_info_cur, learn_info_cur, obs_data_all))  # Learning
                learningOutput[Lidx][Midx][-1][-1]['L2rhoTErr'] = computeL2rhoTErr(learningOutput[Lidx][Midx][-1][-1]['Estimator'], sys_info, obs_info_cur)
                print(f'done ({time.time() - tic_learningroutine:.2f}s).')

            for nidx in range(len(n_plane)):
                learningOutput[Lidx][Midx][nidx][-1]['obs_data']['traj'] = []
                learningOutput[Lidx][Midx][nidx][-1]['obs_data']['x'] = []
                learningOutput[Lidx][Midx][nidx][-1]['obs_data']['xp'] = []
                learningOutput[Lidx][Midx][nidx][-1]['Estimator']['Phi'] = []

        Timings_total[Midx] = time.time() - Timings_total[Midx]
        # Save data
        if save_results_at_every_run:
            print('\n..saving...', end='')
            saving_tic = time.time()
            savemat(os.path.join(SAVE_DIR, f'{sys_info["name"]}_learningOutput{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{os.uname().nodename}(M={obs_info["M"]}).mat'), {
                'sys_info': sys_info,
                'solver_info': solver_info,
                'obs_info': obs_info,
                'learn_info': learn_info,
                'learningOutput': learningOutput,
                'Timings_total': Timings_total,
                'M_plane': M_plane,
                'L_plane': L_plane,
                'n_actual': n_actual
            }, appendmat=False, format='5')
            print(f'done ({time.time() - saving_tic:.2f}s).')

# Results
Err_Rel_EErr = np.zeros((len(L_plane), len(M_plane), len(n_plane), len(learningOutput[0][0]), sys_info['K'], sys_info['K']))
Err_Rel_AErr = np.zeros((len(L_plane), len(M_plane), len(n_plane), len(learningOutput[0][0]), sys_info['K'], sys_info['K']))
Err_Rel_XiErr = np.zeros((len(L_plane), len(M_plane), len(n_plane), len(learningOutput[0][0]), sys_info['K'], sys_info['K']))

for Lidx in range(len(learningOutput)):
    for Midx in range(len(learningOutput[Lidx])):
        for nidx in range(len(learningOutput[Lidx][Midx])):
            for k in range(len(learningOutput[Lidx][Midx][nidx])):
                try:
                    Err_Rel_EErr[Lidx, Midx, nidx, k] = learningOutput[Lidx][Midx][nidx][k]['L2rhoTErr']['EErrSmooth']['Rel']
                except:
                    pass
                try:
                    Err_Rel_AErr[Lidx, Midx, nidx, k] = learningOutput[Lidx][Midx][nidx][k]['L2rhoTErr']['AErrSmooth']['Rel']
                except:
                    pass
                try:
                    Err_Rel_XiErr[Lidx, Midx, nidx, k] = learningOutput[Lidx][Midx][nidx][k]['L2rhoTErr']['XiErrSmooth']['Rel']
                except:
                    pass

# Figures
for k_1 in range(Err_Rel_EErr.shape[4]):
    for k_2 in range(Err_Rel_EErr.shape[5]):
        if np.any(~np.isnan(Err_Rel_EErr[:, :, k_1, k_2])):
            try:
                plt.figure(1)
                plt.subplot(Err_Rel_EErr.shape[4], Err_Rel_EErr.shape[5], k_1 * Err_Rel_EErr.shape[5] + k_2 + 1)
                plt.imshow(np.flipud(np.nanmin(np.mean(Err_Rel_EErr[:, :, k_1, k_2], axis=2), axis=1)), cmap='jet', extent=(L_plane.min(), L_plane.max(), M_plane.min(), M_plane.max()))
                plt.colorbar()
                plt.title(f'Interaction E-kernel ({k_1},{k_2})')
                plt.ylabel('M')
                plt.xlabel('L')

                plt.figure(2)
                plt.subplot(Err_Rel_EErr.shape[4], Err_Rel_EErr.shape[5], k_1 * Err_Rel_EErr.shape[5] + k_2 + 1)
                plt.imshow(np.flipud(np.log10(np.nanmin(np.mean(Err_Rel_EErr[:, :, k_1, k_2], axis=2), axis=1))), cmap='gray', extent=(L_plane.min(), L_plane.max(), M_plane.min(), M_plane.max()))
                plt.colorbar()
                plt.title(f'Interaction E-kernel ({k_1},{k_2})')
                plt.ylabel('M')
                plt.xlabel('L')
            except:
                pass

        if np.any(~np.isnan(Err_Rel_AErr[:, :, k_1, k_2])):
            try:
                plt.figure(3)
                plt.subplot(Err_Rel_EErr.shape[4], Err_Rel_EErr.shape[5], k_1 * Err_Rel_EErr.shape[5] + k_2 + 1)
                plt.imshow(np.flipud(np.nanmin(np.mean(Err_Rel_AErr[:, :, k_1, k_2], axis=2), axis=1)), cmap='jet', extent=(L_plane.min(), L_plane.max(), M_plane.min(), M_plane.max()))
                plt.colorbar()
                plt.title(f'Interaction A-kernel ({k_1},{k_2})')
                plt.ylabel('M')
                plt.xlabel('L')

                plt.figure(4)
                plt.subplot(Err_Rel_EErr.shape[4], Err_Rel_EErr.shape[5], k_1 * Err_Rel_EErr.shape[5] + k_2 + 1)
                plt.imshow(np.flipud(np.log10(np.nanmin(np.mean(Err_Rel_AErr[:, :, k_1, k_2], axis=2), axis=1))), cmap='gray', extent=(L_plane.min(), L_plane.max(), M_plane.min(), M_plane.max()))
                plt.colorbar()
                plt.title(f'Interaction A-kernel ({k_1},{k_2})')
                plt.ylabel('M')
                plt.xlabel('L')
            except:
                pass

        if np.any(~np.isnan(Err_Rel_XiErr[:, :, k_1, k_2])):
            try:
                plt.figure(5)
                plt.subplot(Err_Rel_EErr.shape[4], Err_Rel_EErr.shape[5], k_1 * Err_Rel_EErr.shape[5] + k_2 + 1)
                plt.imshow(np.flipud(np.nanmin(np.mean(Err_Rel_XiErr[:, :, k_1, k_2], axis=2), axis=1)), cmap='jet', extent=(L_plane.min(), L_plane.max(), M_plane.min(), M_plane.max()))
                plt.colorbar()
                plt.title(f'Interaction Xi-kernel ({k_1},{k_2})')
                plt.ylabel('M')
                plt.xlabel('L')

                plt.figure(6)
                plt.subplot(Err_Rel_EErr.shape[4], Err_Rel_EErr.shape[5], k_1 * Err_Rel_EErr.shape[5] + k_2 + 1)
                plt.imshow(np.flipud(np.log10(np.nanmin(np.mean(Err_Rel_XiErr[:, :, k_1, k_2], axis=2), axis=1))), cmap='gray', extent=(L_plane.min(), L_plane.max(), M_plane.min(), M_plane.max()))
                plt.colorbar()
                plt.title(f'Interaction Xi-kernel ({k_1},{k_2})')
                plt.ylabel('M')
                plt.xlabel('L')
            except:
                pass

# SAVING
save_dict = {
    'sys_info': sys_info,
    'solver_info': solver_info,
    'obs_info': obs_info,
    'learn_info': learn_info,
    'Err_Rel_EErr': Err_Rel_EErr,
    'Err_Rel_AErr': Err_Rel_AErr,
    'Err_Rel_XiErr': Err_Rel_XiErr,
    'M_plane': M_plane,
    'L_plane': L_plane
}

savemat(os.path.join(SAVE_DIR, f'{sys_info["name"]}_learningOutputFigErr_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{os.uname().nodename}_.mat'), save_dict, appendmat=False, format='5')

# Done
print('\ndone.\n')
