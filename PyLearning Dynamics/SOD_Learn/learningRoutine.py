
from SOD_Learn.generateObservations import generateObservations
from SOD_Learn.learn_from_dynamics import learn_from_dynamics
from SOD_Learn.estimateRhoLT import estimateRhoLT
from SOD_Utils.regularize_estimator import regularize_estimator

def learningRoutine(solver_info, obs_info, learn_info, obs_data=None):
    learn_out = {}

    if 'VERBOSE' not in learn_info:
        learn_info['VERBOSE'] = 2
    if 'keep_obs_data' not in learn_info:
        learn_info['keep_obs_data'] = True

    # Generate dynamics and observations
    if learn_info['VERBOSE'] > 1:
        print('\n--------------------------------------------------------------------------------')
    if obs_data is None or len(obs_data['x']) == 0:
        print(f"\nGenerating {obs_info['M']} trajectories with {obs_info['L']} observations each...")
        obs_data = generateObservations(learn_info['sys_info'], solver_info, obs_info, obs_info['M'])
        if learn_info['VERBOSE'] > 1:
            print(f"done ({obs_data['Timings']['total']:.2f} sec).")

    # Learn the dynamics from observations
    if learn_info['VERBOSE'] > 1:
        print('\n--------------------------------------------------------------------------------')
        print('Learning...')

    learn_out = learn_from_dynamics(learn_info['sys_info'], obs_info, learn_info, obs_data)
    print("\n ...............Learn from dynamics  is completed.....")
    learn_out['rhoLTemp'] = estimateRhoLT(obs_data, learn_info['sys_info'], obs_info)

    if learn_info['VERBOSE'] >= 1:
        print("done ({} sec).".format(learn_out['Timings']['total']))
        print("The empirical error from learning phis on x/v is: {}.".format(learn_out['Estimator']['emp_err']))
        if 'emp_err_xi' in learn_out['Estimator']:
            print("The empirical error from learning phis on xi is: {}.".format(learn_out['Estimator']['emp_err_xi']))

    # Measure performance in terms of the L2_rho norm
    if learn_info['VERBOSE'] >= 1:
        print('--------------------------------------------------------------------------------')
        print('Regularizing the learned Estimators...')

    # Regularize interaction(s)
    learn_out['Estimator'] = regularize_estimator(learn_out['Estimator'], learn_out['rhoLTemp'], learn_info)

    # Package outputs
    syshat_info = learn_info['sys_info']
    syshat_info['phiE'] = learn_out['Estimator']['phiEhat']
    syshat_info['phiA'] = learn_out['Estimator']['phiAhat']
    syshat_info['phiXi'] = learn_out['Estimator']['phiXihat']
    learn_out['syshat_info'] = syshat_info
    syshatsmooth_info = learn_info['sys_info']
    syshatsmooth_info['phiE'] = learn_out['Estimator']['phiEhatsmooth']
    syshatsmooth_info['phiA'] = learn_out['Estimator']['phiAhatsmooth']
    syshatsmooth_info['phiXi'] = learn_out['Estimator']['phiXihatsmooth']
    learn_out['syshatsmooth_info'] = syshatsmooth_info
    if learn_info['keep_obs_data']:
        learn_out['obs_data'] = obs_data
    else:
        learn_out['obs_data']['Timings'] = obs_data['Timings']

    return learn_out
