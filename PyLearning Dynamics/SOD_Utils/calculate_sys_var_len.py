def calculate_sys_var_len(sys_info):
    # Find out the length of the system variable y = (x, and/or v, and/or xi)^T for a self-organized dynamical system.

    if sys_info['ode_order'] == 1:
        sys_var_len = sys_info['d'] * sys_info['N']  # First order system, contains only x, size (d * N, 1)
    elif sys_info['ode_order'] == 2:
        if sys_info['has_xi']:
            sys_var_len = 2 * sys_info['d'] * sys_info['N'] + sys_info['N']  # Second order system with xi, contains x, v, and xi, size (2 * d * N + N, 1)
        else:
            sys_var_len = 2 * sys_info['d'] * sys_info['N']  # Second order system without xi, contains x and v, size (2 * d * N, 1)

    return sys_var_len
