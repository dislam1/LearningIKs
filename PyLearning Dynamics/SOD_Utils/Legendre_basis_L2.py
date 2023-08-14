import numpy as np

def Legendre_basis_L2(n, x, xspan):
    # The support of the polynomial, [x_k, x_kp1]
    x_k = xspan[0]
    x_kp1 = xspan[1]
    # We need them in increasing order
    if x_kp1 <= x_k:
        raise ValueError("Invalid interval: x_kp1 should be greater than x_k.")
    # Find out the step size
    h_k = x_kp1 - x_k
    # Only find those within the interval [x_k, x_kp1]
    ind = np.where((x_k <= x) & (x < x_kp1))
    # Evaluate
    y, dy = Legendre_poly(n, 2 * (x[ind] - x_k) / h_k - 1)  # Assuming the Legendre_poly function is defined elsewhere
    # Prepare the scaling
    psi = np.zeros_like(x)
    dpsi = np.zeros_like(x)
    # Scale the originals
    psi[ind] = np.sqrt(2 * n + 1) / np.sqrt(h_k) * y
    dpsi[ind] = np.sqrt(2 * n + 1) / np.sqrt(h_k) * dy * 2 / h_k
    return psi, dpsi
