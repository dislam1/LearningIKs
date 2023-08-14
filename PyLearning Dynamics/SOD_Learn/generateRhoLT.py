import os
import numpy as np
from pytictoc import TicToc
from SOD_Learn.generateObservations import generateObservations
from SOD_Learn.estimateRhoLT import estimateRhoLT

def generateRhoLT(sys_info, solver_info, obs_info, reuse_flag=False):
    VERBOSE = obs_info['VERBOSE']
    SAVE_DIR = obs_info['SAVE_DIR']
    filename = os.path.join(SAVE_DIR, sys_info['name'] + '_rhoT.mat')
    
    t=TicToc()
    if not reuse_flag or not os.path.exists(filename):
        tstart = t.tic()
        obs_data = generateObservations(sys_info, solver_info, obs_info, obs_info['M_rhoT'])
        #print('\n')
        #print('obs_data------')
        #print(obs_data)
        #Adding for temporary
        #import pandas as pd
        #rdf = pd.read_excel("D:/temp/M.xls",index_col=0)
        #A = rdf.to_numpy()
        #obs_data['x'] = A
        #End of modification, can be deleted
        print("Observation is completed...moving to estimateRhoLT \n")
        rhoLT = estimateRhoLT(obs_data, sys_info, obs_info)
        rhoLT['Timings'] = t.toc(tstart)
        print("Estimate is completed...moving to estimateRhoLT \n")
        if VERBOSE > 1:
            print(f"\n\tEstimation of rhoLT completed ({rhoLT['Timings']:.2f} seconds).")
        
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        
        np.savez(filename, sys_info=sys_info, solver_info=solver_info, obs_info=obs_info, rhoLT=rhoLT)
    else:
        if VERBOSE > 1:
            print("\n\tgenerateRhoLT loading rhoLT from file")
        
        data = np.load(filename)
        sys_info = data['sys_info'].item()
        solver_info = data['solver_info'].item()
        obs_info = data['obs_info'].item()
        rhoLT = data['rhoLT'].item()
    
    return rhoLT
