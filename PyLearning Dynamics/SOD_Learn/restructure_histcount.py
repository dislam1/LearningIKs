import numpy as np

def restructure_histcount(HCR, HCDR, HCA, HCXi, jHCXi, M, K, type_info, num_bins):
    has_align = bool(HCA)
    has_xi = bool(HCXi)
    new_HCR = np.zeros((K, K, num_bins, M))
    
    if has_align:
        new_HCA = np.zeros((K, K, num_bins, num_bins, M))
        new_HCDR = np.zeros((K, K, num_bins, M))
    
    if has_xi:
        new_jHCXi = np.zeros((K, K, num_bins, num_bins, M))
        new_HCXi = np.zeros((K, K, num_bins, M))
    
    for m in range(M):
        #print('\n from restructure....')
        #print(type(HCR[m]))
        HCR_m = HCR[m]
        
        if has_align:
            HCA_m = HCA[m]
            HCDR_m = HCDR[m]
        
        if has_xi:
            jHCXi_m = jHCXi[m]
            HCXi_m = HCXi[m]
        
        for k1 in range(K):
            num_Ck1 = np.count_nonzero(type_info == (k1 + 1))
            
            for k2 in range(K):
                if k1 == k2:
                    if num_Ck1 == 1:
                        to_update = False
                    else:
                        to_update = True
                else:
                    to_update = True
                
                if to_update:
                    #print(HCR_m[k1,k2])
                    new_HCR[k1, k2, :, m] = HCR_m[k1,k2]   
                    
                    if has_align:
                        new_HCA[k1, k2, :, :, m] = HCA_m[k1][k2]
                        new_HCDR[k1, k2, :, m] = HCDR_m[k1][k2]
                    
                    if has_xi:
                        new_jHCXi[k1, k2, :, :, m] = jHCXi_m[k1][k2]
                        new_HCXi[k1, k2, :, m] = HCXi_m[k1][k2]
    
    output = {'histcountR': new_HCR}
    
    if has_align:
        output['histcountA'] = new_HCA
        output['histcountDR'] = new_HCDR
    
    if has_xi:
        output['jhistcountXi'] = new_jHCXi
        output['histcountXi'] = new_HCXi
    
    return output
