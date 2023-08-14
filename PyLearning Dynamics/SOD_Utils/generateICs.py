import numpy as np
import multiprocessing as mp
from SOD_Utils.calculate_sys_var_len import calculate_sys_var_len
from SOD_Examples.Predators_Swarm.PS_init_config import PS_init_config

#from SOD_Examples.Predators_Swarm.OD_init_config import OD_init_config
#from SOD_Examples.Predators_Swarm.PS_2nd_order_friction import PS_2nd_order_friction
#from SOD_Examples.Predators_Swarm.PT_init_condition import PT_init_condition
#from SOD_EXAMPLES.Predators_Swarm.LJ_init_config import LJ_init_config


def generateICs(M, sys_info):
    # Function to draw a different initial condition for the dynamics
    #def draw_initial_condition():
        #return sys_info["mu0"]()
    
    # Get the length of the initial condition
    #initial_condition_len = len(draw_initial_condition())
    
    # Initialize storage for initial conditions
    y_inits = np.zeros((calculate_sys_var_len(sys_info), M))
    
    # Generate initial conditions in parallel using multiprocessing
    #with mp.Pool() as pool:
        #initial_conditions = pool.map(draw_initial_condition, range(M))
        
    # Fill the y_inits array with the generated initial conditions
    #print(sys_info['mu0'])
    sysa = {} 
    sysa = sys_info
    mu = sys_info['mu0']
    a = mu.split(':')[0]
    if len(a.strip()) > 6 :
        tag = 1  #Parameter
    else:
        tag = 0  #No parameter
    #print(eval(sys_info['mu0'])())
    #for m in range(M):
    m = 0
    while True:
        if tag == 0:
            #No parameter cae
            y_init = eval(sys_info['mu0'])
        else:
            #There are parameter(s)
            #print(y_init)
            y_init = eval(sys_info['mu0'])(sysa)
            if len(y_init)> 0:
                y_inits[:, m] = y_init.ravel()
                m +=1
            if  m== M:
                    break
    
   
    return y_inits
