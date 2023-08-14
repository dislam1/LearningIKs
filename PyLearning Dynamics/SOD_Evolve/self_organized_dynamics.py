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
    #vode',method='adams'

    dynamics = ode(f, jac=None).set_integrator('zvode',method='bdf', atol=10**-6, rtol=10**-6,nsteps=1e8)
    
   
    dynamics.set_initial_value(y_init,t0 ).set_f_params(sys_info,)

    tfinal = solver_info['time_span'][1]   #Final time
    dt = 1.0e-7
    #dt = 0.5

    #tv = np.arange(0,10,0.1)   # Times to evaluate a solution. 
    tv = np.linspace(t0,tfinal,111)
    #tv =np.linspace(0,30,111)
    
    tf = []
    y0 = y_init.reshape((-1,1))
    yf = y0
    #print(yf)

    for i in range(len(tv[1:])):
        dynamics.integrate(dynamics.t+dt)
        if not (dynamics.successful()):
            print('Integration not successful!!')
            dynamics.flag = True
            return
        arr1 = dynamics.y
        arr1 = np.array(arr1)
        arr1 = arr1.reshape((-1,1))
        if (np.real(arr1) == y0).all():
            print('Something wrong with ODE Integration\n')
            dynamics.flag = True
            return

        #print(arr1)
        yf = np.hstack((yf,arr1))
        tf.append(dynamics.t)

    yf = yf[:,1:]
    #print(yf)
    if dynamics.t < solver_info['time_span'][1]:
        flag = False
        stats = {'nfailed':1}
    else:
        flag = True   #Bad data
        stats = {'nfailed':-1}

    
    dynamics.stats = stats

    yR = np.real(yf)
    dynamics.x = yR
    
    dynamics.flag = flag


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