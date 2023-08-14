import numpy as np
from scipy.sparse import spdiags

def backward_differencing_matrix(time_vec):
    # Find out the length of time_vec
    N = len(time_vec)
    # We need time_vec to contain increasing time instances
    assert np.all(np.diff(time_vec) < 0), "time_vec should contain increasing time instances."

    # Find out the time steps
    time_steps = np.zeros(N)
    time_steps[1:] = time_vec[:-1] - time_vec[1:]
    time_steps[0] = time_vec[0] - time_vec[1]

    # Construct backward differentiation matrix
    # On the main diagonal is 1 / \delta_t_i
    main_diag = 1 / time_steps

    # On the sub-diagonal is -1 / \delta_t_{i + 1}
    sub_diag = -1 / time_steps[1:]

    # To use spdiags, sub_diag and main_diag have to have the same size,
    # just add a zero to sub_diag, since it won't be picked
    sub_diag = np.concatenate((sub_diag, [0]))

    # Construct the matrix with spdiags, since it has only two diagonals
    D_mat = spdiags([sub_diag, main_diag], [-1, 0], N, N)

    # D(1, 2) = -1 / \delta_t_1
    D_mat[0, 1] = -1 / time_steps[0]

    return D_mat
