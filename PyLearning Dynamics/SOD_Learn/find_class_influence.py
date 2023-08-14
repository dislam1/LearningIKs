import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from SOD_Utils.standard_basis import standard_basis
from SOD_Utils.Legendre_basis_L2 import Legendre_basis_L2

def find_class_influence(psis, pdist_data, regulation, pdiff_data, d, num_agents, kappa, basis_info, ISSPARSE=False):
    # function class_influence = find_class_influence(psis, pdist_data, regulation, pdiff_data, d, num_agents, kappa)
    #kappa = int(kappa)
    n = len(psis['f'])
    if basis_info.get('is_splines', False):
        
        class_influence = np.zeros((pdiff_data.shape[0], n))
        psi_pdist = []
        dpsi_pdist = []

        for k in range(n):
            psi_of_r, dpsi_of_r = np.array(eval(psis['f'][k])(pdist_data))
            psi_pdist.append(psi_of_r)
            dpsi_pdist.append(dpsi_of_r)
            psi_of_r *= kappa / num_agents
            if regulation is not None:
                psi_of_r *= regulation
            psi_influence = np.kron(psi_of_r, np.ones(d)).reshape(-1, 1) * pdiff_data
            class_influence[:, k] = np.sum(psi_influence, axis=1)
    else:
        num_rows = pdist_data.shape[0]
        num_cols = pdist_data.shape[1]
        maxidx = 0
        """
        Sorted like matlab sort
        EF = A.ravel( order = 'F')
        print(EF)
        EF_SOrted = np.sort(EF)
        print(EF_SOrted)
        C_idx = np.argsort(EF)
        C_idx
        """

        #pdist_data_sorted = np.sort(pdist_data.ravel())
        pdist_data_ravel = pdist_data.ravel( order = 'F')
        pdist_data_sorted = np.sort(pdist_data_ravel)
        pdist_data_sorted_idxs = np.argsort(pdist_data_ravel)
        pdist_counts = np.histogram(pdist_data_sorted, bins=psis['knots'])[0]
        populatedBins = np.where(pdist_counts > 0)[0]
        pdist_counts_cumsum = np.cumsum(pdist_counts)
        sz_class_influence = (pdiff_data.shape[0], len(psis['f']))

        if ISSPARSE:
            i = []
            j = []
            s = []
            i_cur = 0
        else:
            class_influence = np.zeros(sz_class_influence)

        #pdiff_data_d = [pdiff_data[p-1::d, :] for p in range(1, d+1)]
        end = pdiff_data.shape[0]
        pdiff_data_d = [None]*d
        for p in range(d-1, -1, -1) :
            pdiff_data_d_tmp = pdiff_data[p:end:d,:]
            pdiff_data_d_tmp_rav = pdiff_data_d_tmp.ravel(order = 'F')
            pdiff_data_d[p] = np.reshape(pdiff_data_d_tmp_rav[pdist_data_sorted_idxs],pdiff_data_d_tmp.shape )
           
        

        for k in range(1,len(populatedBins)+1):
            if k == 1:
                idxs = np.arange(pdist_counts_cumsum[populatedBins[k-1]])
            else:
                idxs = np.arange(pdist_counts_cumsum[populatedBins[k-2]], pdist_counts_cumsum[populatedBins[k-1]])
            f_idxs = np.where(psis['knotIdxs'] == populatedBins[k-1]+1)[0]

            for kp in range(len(f_idxs)):
                psi_of_r_idxs, dpsi = np.array(eval(psis['f'][f_idxs[kp]])(pdist_data_sorted[idxs]))
                #psi_of_r_idxs = psi_of_r_idxs.ravel()
                psi_of_r_idxs *= kappa / num_agents
                if len(regulation) > 0:
                    reg_wrt_pdist_sorted = regulation[pdist_data_sorted_idxs[idxs]]
                    psi_of_r_idxs *= reg_wrt_pdist_sorted

                idxsI, idxsJ = np.array(np.unravel_index(pdist_data_sorted_idxs[idxs],(num_rows, num_cols)))

                for p in range(d):
                    infl_tmp = psi_of_r_idxs * np.reshape((pdiff_data_d[p].ravel())[idxs],psi_of_r_idxs.shape)
                    #csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
                    infl_tmp_mat = csr_matrix((infl_tmp, (idxsI, idxsJ)), shape=(num_rows, num_cols))
                    if ISSPARSE:
                        itmp = np.arange(p, sz_class_influence[0], d)
                        i.extend(itmp)
                        j.extend([f_idxs[kp]] * len(itmp))
                        s.extend(np.sum(infl_tmp_mat, axis=1))
                        i_cur += len(itmp)
                    else:
                        last = class_influence.shape[0]
                        class_influence[p:last:d, f_idxs[kp]] = (np.sum(infl_tmp_mat, axis=1)).ravel()
                maxidx = max(maxidx, max(f_idxs))

        if ISSPARSE:
            i = np.array(i)
            j = np.array(j)
            s = np.array(s)
            class_influence = csr_matrix((s, (i, j)), shape=sz_class_influence)

    return class_influence
