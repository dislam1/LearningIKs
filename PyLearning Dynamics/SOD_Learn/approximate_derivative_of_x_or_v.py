import numpy as np

def approximate_derivative_of_x_or_v(x, v, time_vec, sys_info):
    #
    # function d_vec = approximate_derivative_of_x_or_v(x, v, time_vec, sys_info, learning_info)
    #
    # (c) Ming Zhong, Mauro Maggioni, JHU

    # find out the number of time instances when observation are made
    L = len(time_vec)
    # it only works when we observe more than twice
    if L == 1:
        raise ValueError('SOD_Learn:approximate_derivative_of_x_or_v:invalidInput', 'Need L > 1!!')
    # find out the number of agents in the system
    N = sys_info['N']
    # find out the size of the state vector for each agent
    d = sys_info['d']
    # find out the order of the ode
    ode_order = sys_info['ode_order']
    # find out either x or v to differentiate
    if ode_order == 1:
        # first order, working only with x
        data = x
    elif ode_order == 2:
        # second order, working only with v
        data = v
    # find out all the time steps
    delta_ts = np.zeros_like(time_vec)
    # t_k - t_{k - 1}
    delta_ts[1:L] = time_vec[1:L] - time_vec[0:(L - 1)]
    # for the initial time, t_1 - t_2
    delta_ts[0] = time_vec[0] - time_vec[1]
    # transpose it to make it a column vector
    delta_ts = np.transpose(delta_ts)
    # make it for every agent data
    delta_ts = np.kron(delta_ts, np.ones((N * d, 1)))
    # the difference between data
    the_diff = np.zeros_like(data)
    # data_k - data_{k - 1}
    the_diff[:, 1:L] = data[:, 1:L] - data[:, 0:(L - 1)]
    # for the data at initial time, data_1 - data_2
    the_diff[:, 0] = data[:, 0] - data[:, 1]
    # make it into a vector
    the_diff = the_diff.flatten()
    # the derivative: (data_k - data_{k - 1})/(t_k - t_{k -1}) except at the
    # initial time: (data_1 - data_2)/(t_1 - t_2)
    d_vec = the_diff / delta_ts
    # for second order, multiply by the mass of each agent
    if ode_order == 2:
        # find out the mass of each agent
        agent_mass = sys_info['agent_mass']
        # we need to find m_i * \dot{v}_i
        # enlarge it to have size N * d for each state vector
        agent_mass = np.kron(agent_mass, np.ones(d))
        # repeat it for all the other time blocks
        agent_mass = np.tile(agent_mass, (L, 1))
        # do m_i * v_i
        d_vec = d_vec * agent_mass
    return d_vec

