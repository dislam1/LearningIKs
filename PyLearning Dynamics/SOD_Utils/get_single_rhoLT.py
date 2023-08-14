def get_single_rhoLT(rhos, k1, k2):
    rho = {
        'histcount': rhos['histcount'][k1][k2],
        'hist': rhos['hist'][k1][k2],
        'supp': rhos['supp'][k1][k2],
        'histedges': rhos['histedges'][k1][k2]
    }
    return rho
