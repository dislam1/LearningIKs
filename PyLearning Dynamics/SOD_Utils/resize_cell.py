def resize_cell(pair_dist, num_classes):
    num_MCs = len(pair_dist)                    # number of Monte Carlo (MC) realizations
    new_pair_dist = [[[] for _ in range(num_MCs)] for _ in range(num_classes)]  # prepare the storage for the new pair_dist

    for MC_ind in range(num_MCs):               # go through each MC realization
        all_pair_dist = pair_dist[MC_ind]       # find out the pair_dist for all classes

        for k_1 in range(num_classes):          # re-distribute them in the new_pair_dist, go through each (C_k1, C_k2) pair
            for k_2 in range(num_classes):
                if MC_ind == 0:                 # when it is the first time, initialize the storage
                    new_pair_dist[k_1][k_2] = [None] * num_MCs

                new_pair_dist[k_1][k_2][MC_ind] = all_pair_dist[k_1][k_2]  # save it

    return new_pair_dist
