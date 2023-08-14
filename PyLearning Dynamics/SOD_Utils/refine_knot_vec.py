def refine_knot_vec(knot_vec, level):
    """
    Refines the given knot vector by increasing the density of sub-intervals.

    Parameters:
        knot_vec (list or numpy array): The original knot vector.
        level (int): The refinement level. It must be an integer greater than or equal to 1.

    Returns:
        new_knot_vec (numpy array): The new knot vector with increased density of sub-intervals.
    """
    import numpy as np

    assert isinstance(knot_vec, (list, np.ndarray)), "knot_vec must be a list or numpy array"
    assert all(np.diff(knot_vec) >= 0), "knot_vec must be non-decreasing"
    assert isinstance(level, int) and level >= 1, "level must be an integer greater than or equal to 1"

    num_knots = len(knot_vec)
    num_sub_int = num_knots - 1
    new_num_sub_int = 2**level * num_sub_int
    new_knot_vec = np.zeros(new_num_sub_int + 1)

    for ind in range(num_knots - 1):
        new_ind = ind + (ind - 1) * (2**level - 1)
        new_knot_vec[new_ind] = knot_vec[ind]
        step_size = knot_vec[ind + 1] - knot_vec[ind]
        new_step_size = step_size * 2**(-level)
        new_inds = np.arange(1, 2**level) + new_ind
        new_knot_vec[new_inds] = new_step_size * np.arange(1, 2**level) + knot_vec[ind]

    new_knot_vec[-1] = knot_vec[-1]
    return new_knot_vec
