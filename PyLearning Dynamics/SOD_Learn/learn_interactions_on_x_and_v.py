import numpy as np
from pytictoc import TicToc
from SOD_Learn.uniform_learn_interactions_on_x_and_v import uniform_learn_interactions_on_x_and_v
from SOD_Utils.LinearCombinationBasis import LinearCombinationBasis


def learn_interactions_on_x_and_v(x, v, xi, dot_xv, time_vec, sys_info, learn_info):
    Estimator = {}
    extra = {}

    t=TicToc()
    #Timings = {}
    Timings = {}
    Timings['LearnInteractions'] = t.tic()

    if learn_info['is_adaptive']:
        Estimator, extra = adaptive_learn_interactions_on_x_and_v(x, v, xi, dot_xv, time_vec, sys_info, learn_info)
    else:
        Estimator, extra = uniform_learn_interactions_on_x_and_v(x, v, xi, dot_xv, time_vec, sys_info, learn_info)

    
    Estimator['Timings']['LearnInteractions'] = t.toc(Timings['LearnInteractions'])

    Estimator['Timings']['LinearCombinationBasis'] = t.tic()
    Estimator['phiEhat'], lastIdx = LinearCombinationBasis(Estimator['Ebasis'], Estimator['alpha'])

    Estimator['phiAhat'] = LinearCombinationBasis(Estimator['Abasis'], Estimator['alpha'][lastIdx + 1:])

    Estimator['Timings']['LinearCombinationBasis'] = t.toc( Estimator['Timings']['LinearCombinationBasis'])

    return Estimator, extra
