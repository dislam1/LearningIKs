import numpy as np


def print_learning_matrix_info(Phi):
    # Print out information about the learning matrix Phi
    print("\nThe following information is about the learning matrix, Phi.")

    # Find out the rank information
    Phi_rank = np.linalg.matrix_rank(Phi)
    print("  The learning matrix, Phi, has rank =", Phi_rank)

    # Find out if Phi has full rank
    num_rows, num_cols = Phi.shape
    use_perc = Phi_rank / num_cols * 100
    print(f"  The learning matrix, Phi, has {num_rows} rows and {num_cols} columns.")
    print(f"  It uses {use_perc:.2f} percentage of columns.")

    # Find out the maximum and minimum singular values of Phi
    sigma_max = np.linalg.norm(Phi)
    cond_num = np.linalg.cond(Phi)
    sigma_min = sigma_max / cond_num
    print(f"  The maximum singular value of Phi is: {sigma_max:12.4e}")
    print(f"  The minimum singular value of Phi is: {sigma_min:12.4e}")
    print(f"  The condition number of Phi is: {cond_num:12.4e}")


# Example usage:
# Assuming you have the learning matrix 'Phi' as a numpy array
# print_learning_matrix_info(Phi)
