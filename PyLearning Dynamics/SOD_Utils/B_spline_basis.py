import numpy as np

def B_spline_basis(x, i, k, knot_vec):
    # Check the parameters
    assert isinstance(i, int) and i > 0, "i should be a positive integer."
    assert isinstance(k, int) and k >= 0, "k should be a non-negative integer."
    assert isinstance(knot_vec, np.ndarray) and np.all(np.diff(knot_vec) >= 0), "Knot vector values should be non-decreasing."

    # Find out the number of elements in the knot vector
    num_elements = len(knot_vec)
    assert num_elements >= k + 2, "Knot vector should have at least {} elements.".format(k + 2)
    assert 1 <= i <= num_elements - k - 1, "Invalid basis index i = {}, expected 1 <= i <= {}".format(i, num_elements - k - 1)

    # The returned value, y, should have the same size as input x
    y = np.zeros_like(x)

    # The first derivative yprime should have the same size as input x
    yprime = np.zeros_like(x)

    # Calculate x_i and x_{i + 1}
    x_i = knot_vec[i - 1]
    x_ip1 = knot_vec[i]

    # Find the B-spline polynomial by its order
    if k == 0:
        # The step function, B-spline
        # The base case for the recursion
        # Do it when x_i is different from x_{i + 1}
        if x_i == x_ip1:
            # Do nothing, y is already initialized to 0
            pass
        else:
            # Find out the interval [x_i, x_{i + 1}), where the i-th B-spline belongs
            indices = np.logical_and(x_i <= x, x < x_ip1)
            # For the first order, B-splines are characteristic functions on the interval [x_i, x_{i + 1})
            y[indices] = 1
            # For k = 0, constant B-spline functions, its derivative is zero
            # Do nothing, yprime is already initialized to 0
    else:
        # Since we are using the same knot vector, simplify
        def B_spline(x, i, k):
            return B_spline_basis(x, i, k, knot_vec)

        # Calculate x_{i + k - 1} and x_{i + k}
        x_ipkp1 = knot_vec[i + k]
        x_ipk = knot_vec[i + k - 1]

        # Do it by cases
        if x_i == x_ipkp1:
            # Do nothing, y is already initialized to 0
            # Do nothing, yprime is already initialized to 0
            pass
        elif x_i < x_ipk and x_ip1 == x_ipkp1:
            # Calculate S_1(x)
            y = (x - x_i) / (x_ipk - x_i) * B_spline(x, i, k - 1)
            # yprime is k * B_{i, k - 1}(x)/(x_{i + k} - x_i)
            yprime = k * B_spline(x, i, k - 1) / (x_ipk - x_i)
        elif x_i == x_ipk and x_ip1 < x_ipkp1:
            # Calculate S_2(x)
            y = (x_ipkp1 - x) / (x_ipkp1 - x_ip1) * B_spline(x, i + 1, k - 1)
            # yprime is -k * B_{i + 1, k - 1}(x)/(x_{i + k + 1} - x_{i + 1})
            yprime = -k * B_spline(x, i + 1, k - 1) / (x_ipkp1 - x_ip1)
        else:
            # Use the B-spline recursion formula
            y = (x - x_i) / (x_ipk - x_i) * B_spline(x, i, k - 1) + (x_ipkp1 - x) / (x_ipkp1 - x_ip1) * B_spline(x, i + 1, k - 1)
            # Use the derivative B-spline recursion formula
            yprime = k * (B_spline(x, i, k - 1) / (x_ipk - x_i) - B_spline(x, i + 1, k - 1) / (x_ipkp1 - x_ip1))

    return y, yprime
