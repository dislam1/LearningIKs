import numpy as np

def find_pair_diff(x):
    """
    Calculates the pairwise differences of observations (agents) in the data matrix x
    
    Args:
    x: The data matrix, with each column of x, i.e., x_i, being a vector of states of agent i.
       x is of size d * N, where d is the size of each state vector and N is the total 
       number of agents (observations)

    Returns:
    pair_diff: Pairwise differences matrix of the form:
               |x_1 - x_1, x_2 - x_1, ..., x_N - x_1|
               |x_1 - x_2, x_2 - x_2, ..., x_N - x_2|
               |   ...    ,    ...    , ...,    ...|
               |x_1 - x_N, x_2 - x_N, ..., x_N - x_N|

    """
    d, N = x.shape
    x_col_vec = x.reshape((d * N, 1))
    x_col_mat = np.tile(x_col_vec, (1, N))
    x_row_mat = np.tile(x, (N, 1))
    pair_diff = x_row_mat - x_col_mat
    
    return pair_diff
