# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:19:43 2023

@author: disla
"""

import numpy as np
import os as sys
#sys.path.append('../SOD_Utils')
from SOD_Utils.partition_sys_var import partition_sys_var
from SOD_Evolve.find_collective_change import find_collective_change
from SOD_Utils.sqdist_mod import sqdist_mod

def eval_rhs(y, sys_info):
    # prepare x, v, xi and pdist_mat (contains the pairwise distance, |x_i - x_i'|)
    #print('From ')
    #print(y)
    state_vars = partition_sys_var(y, sys_info)
    x = state_vars['x']
    v = state_vars['v']
    xi = state_vars['xi']
    #sqdist_mod = x-xi # my modification ** need to check if this works
    pdist_mat = np.sqrt(np.abs(sqdist_mod(x)))  # calculate the pairwise distance (using normal Euclidean distance)

    # evaluate f(t, y) based on the order of the ODE
    if sys_info['ode_order'] == 1:
        rhs = find_collective_change(x, v, xi, pdist_mat, sys_info, 'energy')  # for 1st order system: \dot{x}_i = \sum_{i' = 1}^N \phi^E_{K_i, K_i'}(|x_i - x_i'|)(x_i' - x_i)
    elif sys_info['ode_order'] == 2:
        rhs = np.zeros_like(y)  # 2nd order contains update to x, v (and possibly xi)
        block_size = sys_info['d'] * sys_info['N']
        rhs[:block_size] = v.reshape(-1, 1)  # for the change in x, it is just v
        if sys_info['phiE']:
            internal_energy = find_collective_change(x, v, xi, pdist_mat, sys_info, 'energy')  # collective change from energy based influence
        else:
            internal_energy = np.zeros((block_size, 1))  # zero matrix of size (d * N, 1)
        if sys_info['phiA']:
            internal_alignment = find_collective_change(x, v, xi, pdist_mat, sys_info, 'alignment')  # collective change from alignment based influence
        else:
            internal_alignment = np.zeros((sys_info['d'] * sys_info['N'], 1))  # zero matrix of size (d * N, 1)
        external_change = sys_info['Fv'](v, xi)  # find out the external force
        mass_vec = np.kron(sys_info['agent_mass'], np.ones(sys_info['d']))  # prepare the mass vector for each agent, make it size (d * N, 1)
        rhs[block_size:2 * block_size] = (external_change + internal_energy + internal_alignment) / mass_vec  # m_i \dot{v}_i = F^v + F^E + F^A
        if sys_info['has_xi']:
            internal_change = find_collective_change(x, v, xi, pdist_mat, sys_info, 'xi')  # collective change from energy based influence
            external_change = sys_info['Fxi'](xi, x)  # the non-collective influence on xi
            rhs[2 * block_size:2 * block_size + sys_info['N']] = external_change + internal_change  # \dot{xi}_i = F^\xi + F^xi

    return rhs
