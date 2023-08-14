import numpy as np

def package_histcount_data(HC_info, sys_info, learn_info):
    histcountR = HC_info['histcountR']
    histcountDR = HC_info['histcountDR']
    histcountXi = HC_info['histcountXi']
    histcountA = HC_info['histcountA']
    jhistcountXi = HC_info['jhistcountXi']
    histedgesR = HC_info['histedgesR']
    histedgesDR = HC_info['histedgesDR']
    histedgesXi = HC_info['histedgesXi']
    max_rs = HC_info['max_rs']
    min_rs = HC_info['min_rs']
    max_dotrs = HC_info['max_dotrs']
    min_dotrs = HC_info['min_dotrs']
    max_xis = HC_info['max_xis']
    min_xis = HC_info['min_xis']
    M = len(histcountR)
    has_align = len(histcountDR) > 0
    has_xi = len(histcountXi) > 0

    output = restructure_histcount(histcountR, histcountDR, histcountA, histcountXi, jhistcountXi, M, sys_info['K'], sys_info['type_info'], learn_info['hist_num_bins'])
    histcountR = output['histcountR']

    histcount = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    hist = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    supp = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    histedges = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    max_rs = np.max(max_rs, axis=2)
    min_rs = np.min(min_rs, axis=2)
    histcountR = np.sum(histcountR, axis=3)
    histcount_R = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    hist_R = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    supp_R = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
    for k1 in range(sys_info['K']):
        for k2 in range(sys_info['K']):
            supp_R[k1][k2] = [min_rs[k1][k2], max_rs[k1][k2]]
            histcount_R[k1][k2] = np.squeeze(histcountR[k1][k2])
            hist_R[k1][k2] = histcount_R[k1][k2] / np.sum(histcount_R[k1][k2])

    rhoLTE = {}
    if has_align:
        histcountA = np.sum(histcountA, axis=4)
        histcountDR = np.sum(histcountDR, axis=3)
        max_dotrs = np.max(max_dotrs, axis=2)
        min_dotrs = np.min(min_dotrs, axis=2)
        histcount_DR = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        hist_DR = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        supp_DR = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        for k1 in range(sys_info['K']):
            for k2 in range(sys_info['K']):
                supp[k1][k2] = [min_rs[k1][k2], max_rs[k1][k2], min_dotrs[k1][k2], max_dotrs[k1][k2]]
                histcount[k1][k2] = np.squeeze(histcountA[k1][k2])
                hist[k1][k2] = histcount[k1][k2] / np.sum(np.sum(histcount[k1][k2]))
                histedges[k1][k2] = np.concatenate([histedgesR[k1][k2], histedgesDR[k1][k2]])
                supp_DR[k1][k2] = [min_dotrs[k1][k2], max_dotrs[k1][k2]]
                histcount_DR[k1][k2] = np.squeeze(histcountDR[k1][k2])
                hist_DR[k1][k2] = histcount_DR[k1][k2] / np.sum(histcount_DR[k1][k2])

        rhoLTA = {}
        rhoLTA['histcount'] = histcount
        rhoLTA['hist'] = hist
        rhoLTA['supp'] = supp
        rhoLTA['histedges'] = histedges

        rhoLTR = {}
        rhoLTR['histcount'] = histcount_R
        rhoLTR['hist'] = hist_R
        rhoLTR['supp'] = supp_R
        rhoLTR['histedges'] = histedgesR

        rhoLTDR = {}
        rhoLTDR['histcount'] = histcount_DR
        rhoLTDR['hist'] = hist_DR
        rhoLTDR['supp'] = supp_DR
        rhoLTDR['histedges'] = histedgesDR

        rhoLTA['rhoLTR'] = rhoLTR
        rhoLTA['rhoLTDR'] = rhoLTDR
    else:
        rhoLTA = {}

    rhoLTA = {}
    if has_xi:
        jhistcountXi = np.sum(jhistcountXi, axis=4)
        histcountXi = np.sum(histcountXi, axis=3)
        max_xis = np.max(max_xis, axis=2)
        min_xis = np.min(min_xis, axis=2)
        histcount_Xi = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        hist_Xi = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        supp_Xi = [[None for _ in range(sys_info['K'])] for _ in range(sys_info['K'])]
        for k1 in range(sys_info['K']):
            for k2 in range(sys_info['K']):
                supp[k1][k2] = [min_rs[k1][k2], max_rs[k1][k2], min_xis[k1][k2], max_xis[k1][k2]]
                histcount[k1][k2] = np.squeeze(jhistcountXi[k1][k2])
                hist[k1][k2] = histcount[k1][k2] / np.sum(np.sum(histcount[k1][k2]))
                histedges[k1][k2] = np.concatenate([histedgesR[k1][k2], histedgesXi[k1][k2]])
                supp_Xi[k1][k2] = [min_xis[k1][k2], max_xis[k1][k2]]
                histcount_Xi[k1][k2] = np.squeeze(histcountXi[k1][k2])
                hist_Xi[k1][k2] = histcount_Xi[k1][k2] / np.sum(histcount_Xi[k1][k2])

        rhoLTXi = {}
        rhoLTXi['histcount'] = histcount
        rhoLTXi['hist'] = hist
        rhoLTXi['supp'] = supp
        rhoLTXi['histedges'] = histedges

        rhoLTXi = {}
        rhoLTXi['histcount'] = histcount_R
        rhoLTXi['hist'] = hist_R
        rhoLTXi['supp'] = supp_R
        rhoLTXi['histedges'] = histedgesR

        rhoLTXi = {}
        rhoLTXi['histcount'] = histcount_Xi
        rhoLTXi['hist'] = hist_Xi
        rhoLTXi['supp'] = supp_Xi
        rhoLTXi['histedges'] = histedgesXi

        rhoLTA['rhoLTR'] = rhoLTR
        rhoLTA['rhoLTXi'] = rhoLTXi
    else:
        rhoLTXi = {}

    rhoLT = {}
    rhoLT['rhoLTE'] = rhoLTE
    rhoLT['rhoLTA'] = rhoLTA
    rhoLT['rhoLTXi'] = rhoLTXi

    return rhoLT
