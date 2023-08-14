import numpy as np
from mpi4py import MPI

def parallel_assemble_xi(Rs, x_d, xi_d, dot_xi_d, time_vec, learning_info):
    # Initialization on the client side
    print('Starting from the client side.')
    print('Initializing data for workers.')

    xi_basis_tag = 5
    num_Phi_col_tag = 6

    num_classes = learning_info['K']
    has_derivative = dot_xi_d is not None

    print('Initialization is done.')
    print('All workers get to work!!.')
    print('I will know how many workers there are after I run in parallel.')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_cores_l = comm.Get_size()

    # Get a local copy of x_d and decide if this worker needs to work
    x = x_d
    if x is not None:
        xi = xi_d
        dot_xi = dot_xi_d if has_derivative else None

        has_3D = len(x.shape) == 3
        M = x.shape[2] if has_3D else 1

        if not has_3D:
            print(f'Lab = {rank}: My copy of x has size [{x.shape[0]}, {x.shape[1]}].')
        else:
            print(f'Lab = {rank}: My copy of x has size [{x.shape[0]}, {x.shape[1]}, {M}].')

        pdist_xi_l = [None] * M

        # Build the basis and send it to other workers
        if rank == 0:
            print(f'Lab = {rank}: Building basis.')
            xi_basis_l, xi_knots_l = uniform_basis_on_xi(Rs, learning_info)

            num_Phi_cols = 0
            for k_1 in range(num_classes):
                for k_2 in range(num_classes):
                    num_Phi_cols += len(xi_basis_l[k_1, k_2])

            print(f'Lab = {rank}: The data tags for the different kinds of data are:')
            print(f'  Xi_basis_l:   tag = {xi_basis_tag}.')
            print(f'  Num_Phi_cols: tag = {num_Phi_col_tag}.')

            for lab_ind in range(1, num_cores_l):
                print(f'Lab = {rank}: I am sending 2 pieces of data over, with 2 different tags, to Lab ({lab_ind}).')
                comm.send(xi_basis_l, dest=lab_ind, tag=xi_basis_tag)
                comm.send(num_Phi_cols, dest=lab_ind, tag=num_Phi_col_tag)
        else:
            xi_knots_l = None
            xi_basis_l = comm.recv(source=0, tag=xi_basis_tag)
            num_Phi_cols = comm.recv(source=0, tag=num_Phi_col_tag)

        Phi_l = np.zeros((num_Phi_cols, num_Phi_cols))
        rhs_vec_l = np.zeros((num_Phi_cols, 1))
        rhs_in_l2_norm_sq_l = 0

        # Local Monte Carlo loop
        for m in range(M):
            one_x = x[:, :, m].squeeze()
            one_xi = xi[:, :, m].squeeze()
            one_dot_xi = dot_xi[:, :, m].squeeze() if has_derivative else None

            _, one_pdist_xi, one_Phi, one_F_vec, one_d_vec = one_step_assemble_on_xi(one_x, one_xi, one_dot_xi,
                                                                                      time_vec, xi_basis_l,
                                                                                      learning_info)

            one_rhs_vec = one_d_vec - one_F_vec
            rhs_in_l2_norm_sq_l += np.linalg.norm(one_rhs_vec)**2

            Phi_l += np.transpose(one_Phi) @ one_Phi
            rhs_vec_l += np.transpose(one_Phi) @ one_rhs_vec

            pdist_xi_l[m] = one_pdist_xi

    else:
        num_cores_l = None
        Phi_l = None
        rhs_vec_l = None
        pdist_xi_l = None
        xi_basis_l = None
        xi_knots_l = None
        rhs_in_l2_norm_sq_l = None

    # Gather data from all workers to the client side
    num_cores = comm.gather(num_cores_l, root=0)
    Phi = comm.gather(Phi_l, root=0)
    rhs_vec = comm.gather(rhs_vec_l, root=0)
    pdist_xi = comm.gather(pdist_xi_l, root=0)
    xi_basis = comm.gather(xi_basis_l, root=0)
    xi_knots = comm.gather(xi_knots_l, root=0)
    rhs_in_l2_norm_sq = comm.reduce(rhs_in_l2_norm_sq_l, op=MPI.SUM, root=0)

    if rank == 0:
        Phi = np.sum(Phi, axis=0)
        rhs_vec = np.sum(rhs_vec, axis=0)
        xi_basis = xi_basis[0]
        xi_knots = xi_knots[0]

        pdist_xi = np.concatenate(pdist_xi)

        extra = {'rhs_in_l2_norm_sq': rhs_in_l2_norm_sq}

        print('Back to client side.')
        print('Assembling data.')
        print('Finished!!')

        return xi_knots, xi_basis, pdist_xi, Phi, rhs_vec, extra

    return None

