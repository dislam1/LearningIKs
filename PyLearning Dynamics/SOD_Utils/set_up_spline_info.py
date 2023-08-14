def set_up_spline_info(b, n, num_basis_funs):
    """
    Sets up the necessary fields for the spline_info structure to construct a B-spline basis.

    Parameters:
        b (float): The right end point of the support of B-spline basis functions.
        n (int): The degree of B-spline basis functions.
        num_basis_funs (int): The number of basis functions needed for learning.

    Returns:
        dict: A dictionary containing the necessary information to construct a B-spline basis on [0, b].
    """
    spline_info = {
        'left_end_pt': 0,                  # The left end point of the support of B-spline
        'right_end_pt': b,                 # The right end point of the support of B-spline
        'how_to_create': 'num_basis_first',# The strategy to create the B-spline
        'degree': n,                       # The degree of B-splines (starting from 0)
        'is_clamped': True,                # It has to be a clamped spline
        'num_basis': num_basis_funs,       # The number of basis functions
        'knot_vec': [],                    # Knot vector (empty since we use num_basis_first)
    }
    return spline_info
