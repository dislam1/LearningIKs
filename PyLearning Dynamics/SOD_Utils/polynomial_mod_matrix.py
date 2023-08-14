import numpy as np
from scipy.sparse import spdiags, coo_matrix


def polynomial_mod_matrix(knot_vec, p, type):
    num_knots = len(knot_vec)
    num_subin = num_knots - 1
    num_basis = (p + 1) * num_subin

    if p == 0:
        if num_subin != num_basis:
            raise ValueError('Number of subintervals should be equal to the number of basis functions.')

        if type == 'standard':
            mat1 = spdiags(np.ones(num_subin), 0, num_subin, num_subin)
            mat2 = None
        elif type == 'Legendre':
            steps = knot_vec[1:] - knot_vec[:num_subin]
            main_diag = 1. / np.sqrt(steps)
            mat1 = spdiags(main_diag, 0, num_subin, num_subin)
            mat2 = None
        else:
            raise ValueError('Invalid type.')
    elif p == 1:
        if num_basis != 2 * num_subin:
            raise ValueError('Number of basis functions should be twice the number of subintervals.')

        if type == 'standard':
            entry = np.zeros(4 * num_subin)
            row_ind = np.zeros(4 * num_subin, dtype=int)
            col_ind = np.zeros(4 * num_subin, dtype=int)

            for ind in range(num_subin):
                skip_1 = 4 * ind
                skip_2 = 2 * ind

                entry[skip_1 + 1] = 1
                row_ind[skip_1 + 1] = skip_2 + 1
                col_ind[skip_1 + 1] = row_ind[skip_1 + 1]

                entry[skip_1 + 2] = knot_vec[ind]
                row_ind[skip_1 + 2] = row_ind[skip_1 + 1]
                col_ind[skip_1 + 2] = row_ind[skip_1 + 1] + 1

                entry[skip_1 + 3] = 1
                row_ind[skip_1 + 3] = row_ind[skip_1 + 1] + 1
                col_ind[skip_1 + 3] = row_ind[skip_1 + 1]

                entry[skip_1 + 4] = knot_vec[ind + 1]
                row_ind[skip_1 + 4] = row_ind[skip_1 + 1] + 1
                col_ind[skip_1 + 4] = row_ind[skip_1 + 1] + 1

            mat1 = coo_matrix((entry, (row_ind, col_ind)), shape=(num_basis, num_basis)).tocsr()

            sup_diag = np.ones(num_subin)
            row_ind = np.arange(num_subin)
            col_ind = 2 * row_ind
            mat2 = coo_matrix((sup_diag, (row_ind, col_ind)), shape=(num_subin, num_basis)).tocsr()
        elif type == 'Legendre':
            steps = knot_vec[1:] - knot_vec[:num_subin]
            entry = np.zeros(4 * num_subin)
            row_ind = np.zeros(4 * num_subin, dtype=int)
            col_ind = np.zeros(4 * num_subin, dtype=int)

            for ind in range(num_subin):
                h_k = steps[ind]
                sqrt_h_k = np.sqrt(h_k)
                skip_1 = 4 * ind
                skip_2 = 2 * ind

                entry[skip_1 + 1] = 1 / sqrt_h_k
                row_ind[skip_1 + 1] = skip_2 + 1
                col_ind[skip_1 + 1] = row_ind[skip_1 + 1]

                entry[skip_1 + 2] = -np.sqrt(3) / sqrt_h_k
                row_ind[skip_1 + 2] = row_ind[skip_1 + 1]
                col_ind[skip_1 + 2] = row_ind[skip_1 + 1] + 1

                entry[skip_1 + 3] = entry[skip_1 + 1]
                row_ind[skip_1 + 3] = row_ind[skip_1 + 1] + 1
                col_ind[skip_1 + 3] = row_ind[skip_1 + 1]

                entry[skip_1 + 4] = -entry[skip_1 + 2]
                row_ind[skip_1 + 4] = row_ind[skip_1 + 1] + 1
                col_ind[skip_1 + 4] = row_ind[skip_1 + 1] + 1

            mat1 = coo_matrix((entry, (row_ind, col_ind)), shape=(num_basis, num_basis)).tocsr()

            sup_diag = 2 * np.sqrt(3) / steps ** (3 / 2)
            row_ind = np.arange(num_subin)
            col_ind = 2 * row_ind
            mat2 = coo_matrix((sup_diag, (row_ind, col_ind)), shape=(num_subin, num_basis)).tocsr()
        else:
            raise ValueError('Invalid type.')
    else:
        raise ValueError('Only 0th or 1st degree polynomials are supported!!')

    return mat1, mat2
