import numpy as np
from scipy.sparse import spdiags
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from SOD_Learn.find_class_influence import find_class_influence


L = 10  # Replace with the desired value
total_energy_basis = 256  # Replace with the desired value
N = 10  # Assuming sys_info.N is 5
d = 2  # Assuming sys_info.d is 3

# Calculate the number of non-zero elements
nnz = 10 * L * N * d

# Create a dictionary to store 2D sparse matrices along the third dimension
energy_Phi = []

# Fill in the non-zero elements of the 3D matrix (assuming you have the values)
for k in range(10 * L * N * d):
    # Create a 2D sparse matrix in COO format for each (k) layer
    rows = []  # List of row indices for non-zero elements in this layer
    cols = []  # List of column indices for non-zero elements in this layer
    data = []  # List of data values for non-zero elements in this layer

    for i in range(total_energy_basis):
        # Add the non-zero elements to the respective lists
        rows.append(i)
        cols.append(i)
        data.append(0.0)  # Replace 0.0 with the actual value you want to assign for this layer

    # Create the sparse matrix in COO format for this layer
    energy_Phi.append(sp.coo_matrix((data, (rows, cols)), shape=(total_energy_basis, total_energy_basis)).A)

# Optionally, you can convert individual layers to other sparse formats like CSR or CSC for efficient operations
# For example, to convert the 2nd layer to CSR format:
#energy_Phi_csr_layer_2 = np.array(energy_Phi).tocsr()
sparse_mat = np.array(energy_Phi)
print(sparse_mat.shape)
