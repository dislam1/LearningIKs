from SOD_Utils.set_up_spline_info import set_up_spline_info
from SOD_Utils.construct_B_spline_basis import construct_B_spline_basis
from SOD_Utils.set_up_piecewise_polynomial_info import set_up_piecewise_polynomial_info
from SOD_Utils.construct_piecewise_polynomial_basis import construct_piecewise_polynomial_basis

def uniform_basis(R, degree, num_basis_fun, basis_info):
    if 'is_splines' in basis_info and basis_info['is_splines']:
        spline_info = set_up_spline_info(R, degree, num_basis_fun)
        basis = construct_B_spline_basis(spline_info)
    else:
        polynomial_info = set_up_piecewise_polynomial_info(R, degree, num_basis_fun)
        polynomial_info['type'] = basis_info['type']
        basis = construct_piecewise_polynomial_basis(polynomial_info)
    #print(basis)
    return basis
