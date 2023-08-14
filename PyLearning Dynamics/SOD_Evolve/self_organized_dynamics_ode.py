import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode
from SOD_Evolve.eval_rhs import eval_rhs

def self_organized_dynamics(y_init, sys_info, solver_info):
    """
    Runs a self-organized dynamics of the given form

    Args:
    y_init: Initial condition to run the dynamics
    sys_info: Information regarding the system of the dynamics
    solver_info: Information regarding the setup of the numerical integrator

    Returns:
    dynamics: Structure with `x` and `y` fields representing times and solution values, respectively.
    flag: 0 if integration was successful up to time `solver_info['time_span'][-1]`, 1 otherwise.

    """
    #Normalize the data to avoid dividing zero error
    #med = np.median(y_init[y_init > 0]) #Normalize the initial value if there is any zero.
    #y_init[y_init == 0] = med
    
    # Stochastic dynamics not supported yet
    #print('I am in ode.....\n')
    #print(y_init)
    if sys_info['has_noise']:
        raise NotImplementedError('Stochastic Interaction is not implemented yet!!')
    
    t0 = solver_info['time_span'][0]
    #dynamics = ode(f, jac=None).set_integrator('zvode', method='bdf', with_jacobian=True)

   

    tf = solver_info['time_span'][1]   #Final time
    dt = 1.0e-7
    #dt = 0.0000005
    tv = np.linspace(t0,tf,30)     # Times to evaluate a solution. 
    
    #Commenting to try solve_ivp
    
    #tv =np.linspace(0,30,100)
    yf = []
    tf = []
   
    dynamics = ode(f, jac=None).set_integrator('vode',method='adams')
      
    dynamics.set_initial_value(y_init,t0 ).set_f_params(sys_info,)
    for i in tv[1:]:
        dynamics.integrate(dynamics.t+dt)
        yf.append(dynamics.y)
        tf.append(dynamics.t)
    #print('\nAfter integration')
    #print(yf)
    #while dynamics.successful():

    #print(dynamics.get_return_code())
    #if dynamics.t < solver_info['time_span'][1]:
    if dynamics.t < solver_info['time_span'][1]:
        flag = False
        stats = {'nfailed':1}
    else:
        flag = True   #Bad data
        stats = {'nfailed':-1}

    
    dynamics.stats = stats
    yf=np.array(yf)
    col = yf.shape[0]
    yf = yf.reshape(-1,col)
    #yR = np.fliplr()
    dynamics.x = np.real(yf)
    if yf.shape[0] < 20:
        flag = False
    dynamics.flag = flag


    # Assign time array to dynamics
    #tf=np.array(tf)
    #col = tf.shape[0]
    #tf = tf.reshape(-1,col)
    dynamics.tm = tf
    #Time interval
    dynamics.tv = tv[1:]


    
    return dynamics

def f(t, y, sys_info):
    """
    Nested function to evaluate the right-hand side of the ODE system

    Args:
    t: Current time
    y: Current solution
    sys_info: System information dictionary

    Returns:
    rhs: Right-hand side of the ODE system

    """
    # Extract relevant variables from y
    #print('From odefun \n')
    #print(y)
    #print('\n')
    rhs = eval_rhs(y, sys_info)
   
    return rhs