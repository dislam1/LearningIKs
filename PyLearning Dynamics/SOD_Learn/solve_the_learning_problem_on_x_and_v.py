from SOD_Learn.solve_for_interactions_on_x_and_v_by_others import solve_for_interactions_on_x_and_v_by_others
from SOD_Learn.solve_for_interactions_on_x_and_v_by_CVX import solve_for_interactions_on_x_and_v_by_CVX

def solve_the_learning_problem_on_x_and_v(Phi, rhs_vec, T, M, sys_info, learn_info):
    if learn_info['solver_type'] == 'CVX':
        alpha, fric_coef, opt_val = solve_for_interactions_on_x_and_v_by_CVX(Phi, rhs_vec, T, M, learn_info)
    else:
        alpha, fric_coef, opt_val = solve_for_interactions_on_x_and_v_by_others(Phi, rhs_vec, T, M, learn_info)
    return alpha, fric_coef, opt_val
