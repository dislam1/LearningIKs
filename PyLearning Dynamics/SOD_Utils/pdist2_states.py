import numpy as np

def pdist2_states(states_1, states_2, sys_info):
    def distfun(ZI, ZJ):
        D2 = np.zeros(ZJ.shape[1])
        for row_ind in range(ZJ.shape[1]):
            D2[row_ind] = state_norm(ZI - ZJ[:, row_ind], sys_info)
        return D2

    the_dfunc = lambda ZI, ZJ: distfun(ZI, ZJ)
    D_mat = pdist2(states_1.T, states_2.T, the_dfunc)
    return D_mat

def state_norm(state_diff, sys_info):
    # Calculate the state's norm based on sys_info
    # You need to define how the state's norm is computed based on sys_info
    # For example, if the state is [x; v] (position and velocity), you can use the Euclidean norm.
    # For other types of states, modify this function accordingly.
    # sys_info can provide information on the state type and dimensions to determine the norm.
    pass

def pdist2(X, Y, distfunc):
    # This function calculates the pairwise distance between X and Y using the provided distance function
    # You can use the numpy function np.linalg.norm or any other distance measure in distfunc.
    # distfunc should be a function that takes two input arrays (of the same size) and returns an array of distances.
    pass

# Assuming state_norm and pdist2 functions are implemented correctly based on sys_info.

# Example usage:
# D_mat = pdist2_states(states_1, states_2, sys_info)
