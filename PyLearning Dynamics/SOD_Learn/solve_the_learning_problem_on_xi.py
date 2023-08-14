def solve_the_learning_problem_on_xi(Phi, rhs_vec, basis, T, M, learning_info):
    solver_type = learning_info['xi_solver_type']
    if solver_type == 'CVX':
        alpha_vec, opt_val = solve_for_interactions_on_xi_by_CVX(Phi, rhs_vec, basis, T, M, learning_info)
    else:
        alpha_vec, opt_val = solve_for_interactions_on_xi_by_others(Phi, rhs_vec, basis, T, M, learning_info)
    return alpha_vec, opt_val
