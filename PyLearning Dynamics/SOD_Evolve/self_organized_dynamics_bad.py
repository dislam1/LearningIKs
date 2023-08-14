import numpy as np
from scipy.integrate import solve_ivp
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
    # Stochastic dynamics not supported yet
    if sys_info['has_noise']:
        raise NotImplementedError('Stochastic Interaction is not implemented yet!!')
    tspan = np.linspace(0, 15, 1000)
    t = np.arange(0.0, 40.0, 0.01)
    #print(solver_info)
    ode_opts = {'rtol': solver_info['rel_tol'], 'atol': solver_info['abs_tol']}    # Set ODE solver options
    #dynamics = solve_ivp(lambda t, y: odefun(t, y, sys_info), solver_info['time_span'], y_init, method=solver_info['solver'], **ode_opts)
    #p = (sys_info)
    #rtol = 1e-5
    dynamics = solve_ivp(f, solver_info['time_span'], y_init,  method='LSODA', args=(sys_info,) )
    #from scipy.integrate import ode
    #dynamics = ode(odefun).set_integrator('zvode', method='bdf', with_jacobian=True)
    #dynamics.set_initial_value(y_init, solver_info['time_span']).set_odefun_params(sys_info,)
    #while r.successful() and dynamics.t < solver_info['time_span'][-1]:
    if dynamics.success:
       flag = False
       stats = {'nfailed':1}
    else:
        flag = True   #Bad data
        stats = {'nfailed':-1}
    

    yF = dynamics.y
    if np.isnan(yF).any():
        print("Nan Values from ODE")
        print(yF)
        flag = True

    tv = dynamics.t
    dynamics.x = yF
    dynamics.tv = tv
    dynamics.flag = flag
    dynamics.stats = stats

    

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