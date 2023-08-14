def set_learning_output(obs_data, learn_info, syshat_info, syshatsmooth_info, learning_results, regularization_output, Timings):
    output = {}
    if learn_info['keep_obs_data']:
        output['obs_data'] = obs_data
    else:
        output['obs_data'] = {'Timings': obs_data['Timings']}
        
    output['phiEhat'] = learning_results['phiEhat']
    output['phiEknots'] = learning_results['phiEknots']
    output['phiEhatsmooth'] = regularization_output['phiEhatsmooth']
    output['phiAhat'] = learning_results['phiAhat']
    output['phiAknots'] = learning_results['phiAknots']
    output['phiAhatsmooth'] = regularization_output['phiAhatsmooth']
    output['phiXihat'] = learning_results['phiXihat']
    output['phiXiknots'] = learning_results['phiXiknots']
    output['phiXihatsmooth'] = regularization_output['phiXihatsmooth']
    output['syshat_info'] = syshat_info
    output['syshatsmooth_info'] = syshatsmooth_info
    output['rhoLTemp'] = regularization_output['rhoLTemp']
    output['Timings'] = {
        'L2rhoTE': regularization_output['Timings']['L2rhoTE'],
        'L2rhoTA': regularization_output['Timings']['L2rhoTA'],
        'L2rhoTXi': regularization_output['Timings']['L2rhoTXi'],
        'learn_from_dynamics': Timings['learn_from_dynamics'],
        'estimateRhoLT': Timings['estimateRhoLT']
    }
    return output
