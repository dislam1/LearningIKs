def getAgentInfo(sys_info):
    agents_info = {'idxs': [], 'num_agents': []}
    for k in range(1, sys_info['K'] + 1):
        agents_info['idxs'].append([idx for idx, agent_type in enumerate(sys_info['type_info']) if agent_type == k])
        agents_info['num_agents'].append(len(agents_info['idxs'][k-1]))
    return agents_info
