import numpy as np
from pytictoc import TicToc

def assembleRhoLTemp(histItems, sys_info):
    #
    # (c) M. Zhong (JHU)
    #
    M = len(histItems)

    for m in range(1, M+1):
        pass

    Timings = {'total': None}
    t = TicToc()
    Timings['total'] = t.tic()

    # prepare some indicators
    if sys_info['ode_order'] == 1:
        has_align = False
        has_xi = False
    elif sys_info['ode_order'] == 2:
        has_align = not (sys_info['phiA'] is None)
        has_xi = sys_info['has_xi']

    # initialize storage
    max_rs = np.zeros((sys_info['K'], sys_info['K'], obs_data['x'].shape[2]))
    min_rs = np.zeros((sys_info['K'], sys_info['K'], obs_data['x'].shape[2]))
    max_dotrs, min_dotrs, max_xis, min_xis = [], [], [], []

    if has_align:
        max_dotrs = np.zeros((sys_info['K'], sys_info['K'], obs_data['x'].shape[2]))
        min_dotrs = np.zeros((sys_info['K'], sys_info['K'], obs_data['x'].shape[2]))

    if has_xi:
        max_xis = np.zeros((sys_info['K'], sys_info['K'], obs_data['x'].shape[2]))
        min_xis = np.zeros((sys_info['K'], sys_info['K'], obs_data['x'].shape[2]))

    # use the max_rs, max_dotrs and max_xis from user input if given
    if 'max_rs' in obs_info or obs_info['max_rs'] is not None:
        Mtrajs = obs_data['x']  # not to broadcast the whole obs_data

        # go through each Monte Carlo realization (parfor is not mandatory here)
        for m in range(1, obs_data['x'].shape[2]+1):
            traj = np.squeeze(Mtrajs[:, :, m-1])
            output = find_maximums(traj, sys_info)
            max_rs[:, :, m-1] = output['max_rs']

            if has_align:
                max_dotrs[:, :, m-1] = output['max_dotrs']

            if has_xi:
                max_xis[:, :, m-1] = output['max_xis']

        # find out the maximum over all m realizations
        max_rs = np.max(max_rs, axis=2)

        if has_align:
            max_dotrs = np.max(max_dotrs, axis=2)

        if has_xi:
            max_xis = np.max(max_xis, axis=2)
    else:
        max_rs = obs_info['max_rs']

        if has_align:
            max_dotrs = obs_info['max_dotrs']

        if has_xi:
            max_xis = obs_info['max_xis']

    # prepare the bins for hist count
    histedgesR, histbinwidthR, histedgesDR, histbinwidthDR, histcountR, histcountA, histcountDR, jhistcountXi, histcountXi = \
        prepare_hist_items(sys_info['K'], obs_info['hist_num_bins'], obs_data['x'].shape[2], max_rs, max_dotrs, max_xis)

    # go through each MC realization
    for m in range(1, Mtrajs.shape[2]+1):
        traj = np.squeeze(Mtrajs[:, :, m-1])
        pdist_out = partition_traj(traj, sys_info)
        max_rs[:, :, m-1] = pdist_out['max_r']
        min_rs[:, :, m-1] = pdist_out['min_r']

        histcountR_m = np.empty((sys_info['K'], sys_info['K']), dtype=object)

        if has_align:
            histcountA_m = np.empty((sys_info['K'], sys_info['K']), dtype=object)
            histcountDR_m = np.empty((sys_info['K'], sys_info['K']), dtype=object)
            max_dotrs[:, :, m-1] = pdist_out['max_rdot']
            min_dotrs[:, :, m-1] = pdist_out['min_rdot']

        if has_xi:
            jhistcountXi_m = np.empty((sys_info['K'], sys_info['K']), dtype=object)
            histcountXi_m = np.empty((sys_info['K'], sys_info['K']), dtype=object)
            max_xis[:, :, m-1] = pdist_out['max_xi']
            min_xis[:, :, m-1] = pdist_out['min_xi']

        for k1 in range(1, sys_info['K']+1):
            for k2 in range(1, sys_info['K']+1):
                pdist_x_Ck1_Ck2 = pdist_out['pdist_x'][k1-1][k2-1]

                if pdist_x_Ck1_Ck2.size != 0:
                    histcountR_m[k1-1][k2-1] = np.histogram(pdist_x_Ck1_Ck2.flatten(), bins=histedgesR[k1-1][k2-1], density=False)[0]

                if has_align:
                    pdist_v_Ck1_Ck2 = pdist_out['pdist_v'][k1-1][k2-1]

                    if pdist_v_Ck1_Ck2.size != 0 and pdist_x_Ck1_Ck2.size != 0:
                        histcountA_m[k1-1][k2-1], _, _ = np.histogram2d(pdist_x_Ck1_Ck2.flatten(), pdist_v_Ck1_Ck2.flatten(),
                                                                        bins=[histedgesR[k1-1][k2-1], histedgesDR[k1-1][k2-1]], density=False)
                        histcountDR_m[k1-1][k2-1] = np.histogram(pdist_v_Ck1_Ck2.flatten(), bins=histedgesDR[k1-1][k2-1], density=False)[0]

                if has_xi:
                    pdist_xi_Ck1_Ck2 = pdist_out['pdist_xi'][k1-1][k2-1]

                    if pdist_xi_Ck1_Ck2.size != 0 and pdist_x_Ck1_Ck2.size != 0:
                        jhistcountXi_m[k1-1][k2-1], _, _ = np.histogram2d(pdist_x_Ck1_Ck2.flatten(), pdist_xi_Ck1_Ck2.flatten(),
                                                                          bins=[histedgesR[k1-1][k2-1], histedgesXi[k1-1][k2-1]], density=False)
                        histcountXi_m[k1-1][k2-1] = np.histogram(pdist_xi_Ck1_Ck2.flatten(), bins=histedgesXi[k1-1][k2-1], density=False)[0]

        histcountR[m-1] = histcountR_m

        if has_align:
            histcountA[m-1] = histcountA_m
            histcountDR[m-1] = histcountDR_m

        if has_xi:
            jhistcountXi[m-1] = jhistcountXi_m
            histcountXi[m-1] = histcountXi_m

    # package the data
    rhoLT = package_rhoLT(histedgesR, histcountR, histbinwidthR, histedgesDR, histcountDR, histbinwidthDR,
                          histedgesXi, histcountXi, histbinwidthXi, histcountA, jhistcountXi, Mtrajs.shape[2],
                          sys_info, obs_info, max_rs, min_rs, max_dotrs, min_dotrs, max_xis, min_xis)

    rhoLT['Timings']['total'] = t.toc(Timings['total'])

    return rhoLT
