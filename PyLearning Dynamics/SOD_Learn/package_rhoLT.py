import numpy as np
from SOD_Learn.restructure_histcount import restructure_histcount

def package_rhoLT(histedgesR, histcountR, histbinwidthR, histedgesDR, histcountDR, histbinwidthDR, histedgesXi,
                  histcountXi, histbinwidthXi, histcountA, jhistcountXi, M, sys_info, obs_info, max_rs, min_rs,
                  max_dotrs, min_dotrs, max_xis, min_xis):
    
    if sys_info['ode_order'] == 1:
        has_energy = True
        has_align  =  False
        has_xi  = False
    elif sys_info['ode_order'] == 2:
        if  sys_info['phiE'] is not None:
            has_energy = True
        else:
            has_energy = False
        if sys_info['phiA'] is not None:
            has_align = True
        else:
            has_align = False
        has_xi = sys_info['has_xi']


    output = restructure_histcount(histcountR, histcountDR, histcountA, histcountXi, jhistcountXi, M, sys_info['K'],
                                   sys_info['type_info'], obs_info['hist_num_bins'])
    histcountR = output['histcountR']
    if has_align:
        histcountA = output['histcountA']
        histcountDR = output['histcountDR']
    if has_xi:
        jhistcountXi = output['jhistcountXi']
        histcountXi = output['histcountXi']
    rhoLTE = {}
    histcount = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    hist = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    supp = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    histedges = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    max_rs = np.max(max_rs, axis=2)
    min_rs = np.min(min_rs, axis=2)
    histcountR = np.sum(histcountR, axis=3)
    histcount_R = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    hist_R = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    supp_R = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
    for k1 in range(sys_info['K']):
        for k2 in range(sys_info['K']):
            supp_R[k1][k2] = [min_rs[k1][k2], max_rs[k1][k2]]
            histcount_R[k1][k2] = np.squeeze(histcountR[k1][k2])
            hist_R[k1][k2] = histcount_R[k1][k2] / (np.sum(histcount_R[k1][k2]) * histbinwidthR[k1][k2])
    if has_energy:
        for k1 in range(sys_info['K']):
            for k2 in range(sys_info['K']):
                supp[k1][k2] = supp_R[k1][k2]
                histcount[k1][k2] = histcount_R[k1][k2]
                hist[k1][k2] = hist_R[k1][k2] 
  
        rhoLTE['histcount'] = histcount
        rhoLTE['hist'] = hist
        rhoLTE['supp'] = supp
        rhoLTE['histedges'] = histedgesR
    else:
        rhoLTE = None

    rhoLTA = {}
    if has_align:
        histcountA = np.sum(histcountA, axis=4)
        histcountDR = np.sum(histcountDR, axis=3)
        max_dotrs = np.max(max_dotrs, axis=2)
        min_dotrs = np.min(min_dotrs, axis=2)
        histcount_DR = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
        hist_DR = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
        supp_DR = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
        for k1 in range(sys_info['K']):
            for k2 in range(sys_info['K']):
                supp[k1][k2] = [min_rs[k1][k2], max_rs[k1][k2], min_dotrs[k1][k2], max_dotrs[k1][k2]]
                histcount[k1][k2] = np.squeeze(histcountA[k1][k2])
                hist[k1][k2] = histcount[k1][k2] / (
                            np.sum(np.sum(histcount[k1][k2])) * histbinwidthR[k1][k2] * histbinwidthDR[k1][k2])
                histedges[k1][k2] = np.concatenate((histedgesR[k1][k2], histedgesDR[k1][k2]))
                supp_DR[k1][k2] = [min_dotrs[k1][k2], max_dotrs[k1][k2]]
                histcount_DR[k1][k2] = np.squeeze(histcountDR[k1][k2])
                hist_DR[k1][k2] = histcount_DR[k1][k2] / (np.sum(histcount_DR[k1][k2]) * histbinwidthDR[k1][k2])

     
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
        rhoLTA = None

    rhoLTXi = {}
    if has_xi:
        jhistcountXi = np.sum(jhistcountXi, axis=4)
        histcountXi = np.sum(histcountXi, axis=3)
        max_xis = np.max(max_xis, axis=2)
        min_xis = np.min(min_xis, axis=2)
        histcount_Xi = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
        hist_Xi = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
        supp_Xi = [[None] * sys_info['K'] for _ in range(sys_info['K'])]
        for k1 in range(sys_info['K']):
            for k2 in range(sys_info['K']):
                supp[k1][k2] = [min_rs[k1][k2], max_rs[k1][k2], min_xis[k1][k2], max_xis[k1][k2]]
                histcount[k1][k2] = np.squeeze(jhistcountXi[k1][k2])
                hist[k1][k2] = histcount[k1][k2] / (
                            np.sum(np.sum(histcount[k1][k2])) * histbinwidthR[k1][k2] * histbinwidthXi[k1][k2])
                histedges[k1][k2] = np.concatenate((histedgesR[k1][k2], histedgesXi[k1][k2]))
                supp_Xi[k1][k2] = [min_xis[k1][k2], max_xis[k1][k2]]
                histcount_Xi[k1][k2] = np.squeeze(histcountXi[k1][k2])
                hist_Xi[k1][k2] = histcount_Xi[k1][k2] / (np.sum(histcount_Xi[k1][k2]) * histbinwidthXi[k1][k2])

        rhoLTXi = {}
        rhoLTXi['histcount'] = histcount
        rhoLTXi['hist'] = hist
        rhoLTXi['supp'] = supp
        rhoLTXi['histedges'] = histedges

        rhoLTR = {}
        rhoLTR['histcount'] = histcount_R
        rhoLTR['hist'] = hist_R
        rhoLTR['supp'] = supp_R
        rhoLTR['histedges'] = histedgesR

        rhoLTXi = {}
        rhoLTXi['histcount'] = histcount_Xi
        rhoLTXi['hist'] = hist_Xi
        rhoLTXi['supp'] = supp_Xi
        rhoLTXi['histedges'] = histedgesXi

        rhoLTXi['rhoLTR'] = rhoLTR
        rhoLTXi['rhoLTDR'] = rhoLTXi
    else:
        rhoLTXi = None

    rhoLT = {}
    rhoLT['rhoLTE'] = rhoLTE
    rhoLT['rhoLTA'] = rhoLTA
    rhoLT['rhoLTXi'] = rhoLTXi
    return rhoLT
