import numpy as np

def Legendre_poly(n, x):
    # where x is in [-1, 1]
    ind = (-1 <= x) & (x < 1)
    # the polynomial value
    y = np.zeros_like(x)
    # the derivative of the polynomial
    dy = np.zeros_like(x)
    # do it by case, the degree of the Legendre polynomial
    if n == 0:
        P = np.array([1])
        dP = 0
    elif n == 1:
        P = np.array([1, 0])
        dP = np.array([1])
    elif n == 2:
        P = np.array([3/2, 0, -1/2])
        dP = np.array([3, 0])
    elif n == 3:
        P = np.array([5/2, 0, -3/2, 0])
        dP = np.array([15/2, 0, -3])
    elif n == 4:
        P = np.array([35/8, 0, -30/8, 0, 3/8])
        dP = np.array([35/4, 0, -15/2, 0])
    elif n == 5:
        P = np.array([63/8, 0, -70/8, 0, 15/2, 0])
        dP = np.array([315/8, 0, -210/2, 0, 15])
    elif n == 6:
        P = np.array([231/16, 0, -315/16, 0, 105/16, 0, -5/16])
        dP = np.array([693/16, 0, -630/16, 0, 105/16, 0])
    elif n == 7:
        P = np.array([429/16, 0, -693/16, 0, 315/16, 0, -35/16, 0])
        dP = np.array([3003/16, 0, -3465/16, 0, 945/16, 0, -35/16])
    elif n == 8:
        P = np.array([6435/128, 0, -12012/128, 0, 6930/128, 0, -1260/128, 0, 35/128])
        dP = np.array([6435/64, 0, -6006/64, 0, 6930/64, 0, -2520/64, 0])
    elif n == 9:
        P = np.array([12155/128, 0, -25740/128, 0, 18018/128, 0, -4620/128, 0, 315/128, 0])
        dP = np.array([109395/128, 0, -77220/128, 0, 36036/128, 0, -4620/128, 0, 315/128])
    elif n == 10:
        P = np.array([46189/256, 0, -109395/256, 0, 90090/256, 0, -30030/256, 0, 3465/256, 0, -63/256])
        dP = np.array([209715/256, 0, -328185/256, 0, 225225/256, 0, -90090/256, 0, 6930/256, 0])
    else:
        raise ValueError("Only Legendre polynomials up to degree 10 are supported.")
    # return the values
    y[ind] = np.polyval(P, x[ind])
    dy[ind] = np.polyval(dP, x[ind])
    return y, dy
