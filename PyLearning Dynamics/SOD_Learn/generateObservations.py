import numpy as np
from pytictoc import TicToc
from SOD_Utils.calculate_sys_var_len import calculate_sys_var_len
from SOD_Utils.generateICs import generateICs
from SOD_Utils.observe_dynamics import observe_dynamics
from SOD_Evolve.self_organized_dynamics import self_organized_dynamics


def generateObservations(sys_info, solver_info, obs_info, M):
    #Define obs_data dictionary to return
    obs_data = {}
    obs_data['Timings']={}
    t=TicToc()
    obs_data['Timings']['total'] = t.tic()
    if M is None:
        M = obs_info['M']
    #print(obs_data) 
    # obs_data=2000
    
    comp_ICs = 'ICs' not in obs_data or obs_data['ICs'] is None
    comp_traj = 'traj' not in obs_data or obs_data['traj'] is None
    
    sys_var_len = calculate_sys_var_len(sys_info)
    x_obs = np.zeros((sys_var_len, obs_info['L'], M))
    
    if obs_info['use_derivative']:
        xp_obs = np.zeros((sys_var_len, obs_info['L'], M))
        
    flag = np.zeros(M, dtype=bool)
    
    if comp_ICs:
        obs_data['Timings']['ICs'] = t.tic()
        ICs = generateICs(M, sys_info)
        obs_data['Timings']['ICs'] = t.toc(obs_data['Timings']['ICs'])
    else:
        ICs = obs_data['ICs']  
    
    Timingsobsdynamics = np.zeros(M)
    
    if not comp_traj:
        traj = obs_data['traj']
    else:
        traj = [None] * M
    #Check for zero diving error
    # If array has Nan or inf, ignore
    #ICs.reshape(M,-1)
    #print('from Objectgeneration')
    print(ICs.shape)
    print('---------------------\n')
    for m in range(M):
        if comp_traj:
            #print(ICs[m,:])
            traj[m] = self_organized_dynamics(ICs[m,:], sys_info, solver_info)

        
        flag[m] = traj[m].flag
        if flag[m]:
            continue
        #Checking y is Nan or Inf
        y_flag = 1
        y = traj[m].x
        if y.shape[0] < 10:
            continue

        ind = np.isinf(y)
        if np.sum(ind) > 0:
            y_flag = 0
        ind = np.isnan(y)
        if np.sum(ind) > 0:
            y_flag = 0
        if y_flag == 0:
            flag[m] = True
            continue
        x_obs[:, :, m], dobstraj, _, Timingsobsdynamics[m] = observe_dynamics(traj[m], obs_info)
        
        if obs_info['use_derivative']:
            xp_obs[:, :, m] = dobstraj
    #-----End of for loop--------

    obs_data['nFail'] = np.sum(flag)
    obs_data['ICs_fail'] = ICs[flag]
    ICs[:, flag]         = None
    obs_data['ICs']      = ICs
        
    traj = [traj[m] for m in range(M) if not flag[m]]
    obs_data['traj'] = traj
    x_obs[:, :, flag] = None
    obs_data['x'] = x_obs
    
    if obs_info['use_derivative']:
        #xp_obs = xp_obs[:, :, ~flag]
        obs_data['xp'] = xp_obs
        obs_data['xp'][:,:,flag] = None
    
    obs_data['Timings']['obsdynamics'] = Timingsobsdynamics
    obs_data['Timings']['total'] = t.toc(obs_data['Timings']['total'])
   
    
    return obs_data
