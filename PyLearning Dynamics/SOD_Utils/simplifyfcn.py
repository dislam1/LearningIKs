import numpy as np
from scipy import interpolate
from scipy.interpolate import NearestNDInterpolator

from SOD_Utils.eval_basis_functions import eval_basis_functions

def simplifyfcn(f, interpts, interval, degree):
    interpts_sub = interpts[(interpts >= interval[0]) & (interpts <= interval[1])]
    evalpts = (interpts_sub[:-1] + interpts_sub[1:]) / 2

    alpha_vec = f['alpha_vec']
    basis = f['basis']    
    fun = f['fun']
    
        
    try:
        if degree == 0:
            fvals, fprime = (lambda r : eval_basis_functions(r, alpha_vec, basis) )(evalpts)
            f_simple = interpolate.interp1d(evalpts, fvals, kind='nearest',bounds_error=False, fill_value='extrapolate')
            #f_simple = interpolate.interp1d(evalpts, fvals, kind='nearest')
            #f_simple = NearestNDInterpolator(evalpts, fvals)
            #f_simple = 'lambda r:'+str(f_inter)+'(r)'
        else:
            fvals, fprime = (lambda r : eval_basis_functions(r, alpha_vec, basis) )(evalpts)
            f_simple = interpolate.interp1d(evalpts, fvals, bounds_error=False, kind='linear',fill_value='extrapolate')
            #f_simple = interpolate.interp1d(evalpts, fvals, bounds_error=False, kind='linear')
            #f_simple = 'lambda r:'+str(f_inter)+'(r)'
    except:
        print('Eception - from simplyfyfcn\n')
        f_simple = fun

    return f_simple
