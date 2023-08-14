import numpy as np

def insert_time_vec(time_vec, T):
    # Find the index for the insertion time T
    ind = np.argmax(time_vec >= T)

    # Check if T is already in time_vec
    if time_vec[ind] == T:
        # If T is already in time_vec, return time_vec as is
        time_vec_new = time_vec
    else:
        # If T is not in time_vec, insert it into the time_vec
        time_vec_new = np.insert(time_vec, ind, T)

    return time_vec_new
