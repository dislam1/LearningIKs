import numpy as np
from SOD_Utils.standard_basis import standard_basis
from SOD_Utils.Legendre_basis_L2 import Legendre_basis_L2

def construct_piecewise_polynomial_basis(polynomial_info):
    degree = polynomial_info['degree']
    basis = {}
    
    if polynomial_info['how_to_create'] == 'num_basis_first':
        a = int(polynomial_info['left_end_pt'])
        b = polynomial_info['right_end_pt']
        n = int(polynomial_info['num_basis'])
        print('From construct_piecewise_polynomial_basis a,b, n\n')
        print(a,b,n)
        #Add 0.25 to b to adjust
        
        if n % (degree + 1) != 0:
            raise ValueError('The total number of basis functions has to be a multiple of degree + 1.')

        num_sub_inter = n // (degree + 1)
        num_knots = num_sub_inter + 1

        basis['knots'] = np.linspace(a, b, num_knots.astype(int))
        basis['f'] = [None]*n
        basis['knotIdxs'] = np.zeros(n, dtype=np.uint32)
        
        for ind in range(1, n+1):
            ell = ind - 1
            n = ell % (degree + 1)
            k = (ell - n) // (degree + 1) + 1
            basis['knotIdxs'][ind - 1] = k
            xspan = [basis['knots'][k-1], basis['knots'][k]]
            if polynomial_info['type'] == 'Legendre':
                psi = 'lambda r, n='+str(n)+', xspan='+str(xspan)+': Legendre_basis_L2(n, r, xspan)'
                #psi = 'lambda r : Legendre_basis_L2(n, r, xspan)'
                basis['f'][ind - 1] = psi
            elif polynomial_info['type'] == 'standard':
                psi = 'lambda r, n='+str(n)+', xspan='+str(xspan)+': standard_basis(n, r, xspan)'
                basis['f'][ind - 1] = psi
            else:
                raise ValueError('Only Legendre and standard polynomials are supported for now.')

    elif polynomial_info['how_to_create'] == 'basis.knots_first':
        basis['knots'] = polynomial_info['basis']['knots']
        num_knots = len(basis['knots'])
        num_sub_inter = num_knots - 1
        n = num_sub_inter * (degree + 1)
        basis['f'] = []*n
        basis['knotIdxs'] = np.zeros(n, dtype=np.uint32)

        for ind in range(1, n+1):
            ell = ind - 1
            n = ell % (degree + 1)
            k = (ell - n) // (degree + 1) + 1
            basis['knotIdxs'][ind - 1] = k
            xspan = [basis['knots'][k - 1], basis['knots'][k]]
            if polynomial_info['type'] == 'Legendre':
                psi = 'lambda r, n='+str(n)+', xspan='+str(xspan)+': Legendre_basis_L2(n, r, xspan)'
                basis['f'][ind - 1] = psi
            elif polynomial_info['type'] == 'standard':
                psi = 'lambda r, n='+str(n)+', xspan='+str(xspan)+': standard_basis(n, r, xspan)'
                basis['f'][ind - 1] = psi
            else:
                raise ValueError('Only Legendre and standard polynomials are supported for now.')
    
    return basis
