import numpy as np

def construct_B_spline_basis(spline_info):
    p = spline_info['degree']
    if p < 0:
        raise ValueError("The degree for B-spline basis has to be non-negative!")

    if spline_info['is_clamped']:
        if spline_info['how_to_create'] == 'num_basis_first':
            a = spline_info['left_end_pt']
            b = spline_info['right_end_pt']
            D_of_N = spline_info['num_basis']
            basic_knot = np.linspace(a, b, D_of_N - p + 1)
            if p > 0:
                knot_vec = np.concatenate([np.full(p, a), basic_knot, np.full(p, b)])
            else:
                knot_vec = basic_knot
        elif spline_info['how_to_create'] == 'knot_vec_first':
            knot_vec = spline_info['knot_vec']
            num_knots = len(knot_vec)
            if num_knots < 2:
                raise ValueError("The number of knots in the given knot vector has to be at least 2!")
            if not np.all(np.diff(knot_vec) >= 0):
                raise ValueError("The knot vector must have non-decreasing knots!")
            a, b = knot_vec[0], knot_vec[-1]
            basic_knot = knot_vec
            if p > 0:
                knot_vec = np.concatenate([np.full(p, a), basic_knot, np.full(p, b)])
            else:
                knot_vec = basic_knot
            D_of_N = num_knots - p - 1
    else:
        if spline_info['how_to_create'] == 'num_basis_first':
            a = spline_info['left_end_pt']
            b = spline_info['right_end_pt']
            D_of_N = spline_info['num_basis']
            num_knots = D_of_N + p + 1
            knot_vec = np.linspace(a, b, num_knots)
        elif spline_info['how_to_create'] == 'knot_vec_first':
            knot_vec = spline_info['knot_vec']
            num_knots = len(knot_vec)
            if num_knots < 2:
                raise ValueError("The number of knots in the given knot vector has to be at least 2!")
            if not np.all(np.diff(knot_vec) >= 0):
                raise ValueError("The knot vector must have non-decreasing knots!")
            D_of_N = num_knots - p - 1

    basis_funs = [lambda x, l=l, p=p, knot_vec=knot_vec: B_spline_basis(x, l, p, knot_vec) for l in range(1, D_of_N + 1)]
    return {'f': basis_funs, 'knots': knot_vec, 'knotIdxs': np.arange(1, len(knot_vec))}
