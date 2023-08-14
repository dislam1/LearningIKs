from pytictoc import TicToc

def learn_interactions_on_xi(Rs, x, xi, dot_xi, time_vec, sys_info, learn_info):
    Estimator = {}
    extra = {}
    t=TicToc()
    Timings = {}
    Timings['LearnInteractions'] = t.tic()

    if learn_info['is_adaptive']:
        Estimator, extra = adaptive_learn_interactions_on_xi(Rs, x, xi, dot_xi, time_vec, sys_info, learn_info)
    else:
        Estimator, extra = uniform_learn_interactions_on_xi(Rs, x, xi, dot_xi, time_vec, sys_info, learn_info)

    Estimator['Timings'] = {}
    Estimator['Timings']['LearnInteractions'] = t.toc() - Timings['LearnInteractions']

    Estimator['Timings']['LinearCombinationBasis'] = t.tic()
    Estimator['phiXihat'] = LinearCombinationBasis(Estimator['Xibasis'], Estimator['alpha'])
    Estimator['Timings']['LinearCombinationBasis'] = t.toc() - Estimator['Timings']['LinearCombinationBasis']

    return Estimator, extra
