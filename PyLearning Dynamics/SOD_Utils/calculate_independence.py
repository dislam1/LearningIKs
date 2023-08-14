import numpy as np

def calculate_independence(jhist, mhist1, mhist2, bwidth1, bwidth2):
    # Transpose mhist1 and mhist2 if they are not in the required shape
    mhist1 = mhist1.reshape(-1, 1) if mhist1.ndim == 1 else mhist1
    mhist2 = mhist2.reshape(1, -1) if mhist2.ndim == 1 else mhist2

    # Calculate the difference between jhist and mhist1 * mhist2
    the_diff = np.sum(np.abs(jhist - np.dot(mhist1, mhist2))) * bwidth1 * bwidth2

    return the_diff
