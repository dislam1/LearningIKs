import numpy as np

def find_maximums(traj, sys_info):
    #print(traj)
    L = traj.shape[1]
    rs_each_type = np.zeros(sys_info['K'])
    one_block = sys_info['N'] * sys_info['d']
    agent_each_type = np.zeros((sys_info['K'], sys_info['N']), dtype=bool)
    
    if sys_info['ode_order'] == 2:
        has_align = bool(sys_info['phiA'])
        has_xi = sys_info['has_xi']
    else:
        has_align = False
        has_xi = False
        
    if has_align:
        dotrs_each_type = np.zeros(sys_info['K'])
        
    if has_xi:
        xis_each_type = np.zeros(sys_info['K'])
        
    for l in range(L):
        x_at_tl = traj[:one_block, l].reshape((sys_info['d'], sys_info['N']))
        
        if has_align:
            v_at_tl = traj[one_block:2 * one_block, l].reshape((sys_info['d'], sys_info['N']))
        
        if has_xi:
            xi_at_tl = traj[2 * one_block:2 * one_block + sys_info['N'], l]
        
        rs_at_tl = np.sqrt((np.sum(x_at_tl ** 2, axis=0)).astype(float))
        
        if has_align:
            dotrs_at_tl = np.sqrt(np.sum(v_at_tl ** 2, axis=0))
        
        if has_xi:
            xis_at_tl = np.sqrt(np.sum(xi_at_tl ** 2, axis=0))
        
        for k in range(sys_info['K']):
            if l == 0:
                #agent_Ck = 1 if sys_info['type_info'] == k else 0
                # Corrected to get the list 
                agent_Ck = sys_info['type_info'] ==k+1
                agent_each_type[k, :] = agent_Ck
            else:
                agent_Ck = agent_each_type[k, :]
            #print(rs_at_tl[agent_Ck])    
            rs_each_type[k] = np.max(rs_at_tl[agent_Ck])
            
            if has_align:
                #print(dotrs_at_tl[agent_Ck])
                dotrs_each_type[k] = np.max(dotrs_at_tl[agent_Ck])
            
            if has_xi:
                print(xis_at_tl[agent_Ck])
                xis_each_type[k] = np.max(xis_at_tl[agent_Ck])
    
        if l == 0:
            max_rs = np.tile(rs_each_type, (sys_info['K'], 1)) + np.tile(rs_each_type.reshape((-1, 1)), (1, sys_info['K']))
            
            if has_align:
                max_dotrs = np.tile(dotrs_each_type, (sys_info['K'], 1)) + np.tile(dotrs_each_type.reshape((-1, 1)), (1, sys_info['K']))
            
            if has_xi:
                max_xis = np.tile(xis_each_type, (sys_info['K'], 1)) + np.tile(xis_each_type.reshape((-1, 1)), (1, sys_info['K']))
        else:
            a_max = np.tile(rs_each_type.ravel(), (sys_info['K'], 1)) + np.tile(rs_each_type.reshape((-1, 1)), (1, sys_info['K']))
            max_rs = np.maximum(max_rs, a_max)
            
            if has_align:
                a_max = np.tile(dotrs_each_type.ravel(), (sys_info['K'], 1)) + np.tile(dotrs_each_type.reshape((-1, 1)), (1, sys_info['K']))
                max_dotrs = np.maximum(max_dotrs, a_max)
            
            if has_xi:
                a_max = np.tile(xis_each_type.ravel(), (sys_info['K'], 1)) + np.tile(xis_each_type.reshape((-1, 1)), (1, sys_info['K']))
                max_xis = np.maximum(max_xis, a_max)
    
    output = {"max_rs": max_rs}
    
    if has_align:
        output["max_dotrs"] = max_dotrs
    
    if has_xi:
        output["max_xis"] = max_xis
    
    return output
