import numpy as np

def generateGravityMat(learningOutput, method):
    if method == 1 or method == 2:
        sunEstimators = learningOutput[0].Estimator.phiEhat[1:, 0]
        rhoLTemps = {}
        rhoLTemps['hist'] = learningOutput[0].rhoLTemp.rhoLTE.hist[1:, 0]
        rhoLTemps['histedges'] = learningOutput[0].rhoLTemp.rhoLTE.histedges[1:, 0]
        rhoLTemps['supp'] = learningOutput[0].rhoLTemp.rhoLTE.supp[1:, 0]
        
        num_planets = len(sunEstimators)
        refine_level = num_planets
        num_pts = 4 ** (refine_level + 2) - 1
        knots = np.zeros(num_planets * num_pts)
        
        for ind in range(num_planets):
            ind1 = (ind - 1) * num_pts
            ind2 = ind * num_pts
            knots[ind1:ind2] = np.linspace(rhoLTemps['supp'][ind][0], rhoLTemps['supp'][ind][1], num_pts)
        
        rp = np.unique(knots)
        P = len(rp)
        
        sunPhiMat = np.zeros((num_planets, P))
        sunRhoMat = np.zeros((num_planets, P))
        
        for ind in range(num_planets):
            sunPhiMat[ind, :] = sunEstimators[ind](rp)
            sunRhoMat[ind, :] = evaluate_rhoLT(rhoLTemps['hist'][ind], rhoLTemps['histedges'][ind], rhoLTemps['supp'][ind], rp)
        
        gravity_terms = {}
        gravity_terms['Phii1Mat'] = sunPhiMat
        gravity_terms['Rhoi1Mat'] = sunRhoMat
        gravity_terms['rp'] = rp
        
        planetEstimators = learningOutput[0].Estimator.phiEhat[0, 1:]
        rhoLTemps = {}
        rhoLTemps['hist'] = learningOutput[0].rhoLTemp.rhoLTE.hist[0, 1:]
        rhoLTemps['histedges'] = learningOutput[0].rhoLTemp.rhoLTE.histedges[0, 1:]
        rhoLTemps['supp'] = learningOutput[0].rhoLTemp.rhoLTE.supp[0, 1:]
        
        planetPhiMat = np.zeros((num_planets, P))
        planetRhoMat = np.zeros((num_planets, P))
        
        for ind in range(num_planets):
            planetPhiMat[ind, :] = planetEstimators[ind](rp)
            planetRhoMat[ind, :] = evaluate_rhoLT(rhoLTemps['hist'][ind], rhoLTemps['histedges'][ind], rhoLTemps['supp'][ind], rp)
        
        gravity_terms['Phi1iMat'] = planetPhiMat
        gravity_terms['Rho1iMat'] = planetRhoMat
        
    elif method == 3:
        N = learningOutput[0].Estimator.phiEhat.shape[0]
        num_pts = 2 ** N - 1
        knots = np.zeros(((N ** 2 - N) // 2) * num_pts)
        N_sums = np.cumsum([0] + list(range(N - 1, 1, -1)))
        
        for i_ind in range(N):
            for j_ind in range(i_ind + 1, N):
                supp = learningOutput[0].rhoLTemp.rhoLTE.supp[i_ind, j_ind]
                ind = N_sums[i_ind] + (j_ind - i_ind)
                ind1 = (ind - 1) * num_pts
                ind2 = ind * num_pts
                knots[ind1:ind2] = np.linspace(supp[0], supp[1], num_pts)
        
        rp = np.unique(knots)
        P = len(rp)
        
        PhiMat = np.zeros((N, N, P))
        RhoMat = np.zeros((N, N, P))
        
        for i_ind in range(N):
            for j_ind in range(i_ind + 1, N):
                PhiMat[i_ind, j_ind, :] = learningOutput[0].Estimator.phiEhat[i_ind, j_ind](rp)
                PhiMat[j_ind, i_ind, :] = learningOutput[0].Estimator.phiEhat[j_ind, i_ind](rp)
                
                hist = learningOutput[0].rhoLTemp.rhoLTE.hist[i_ind, j_ind]
                histedges = learningOutput[0].rhoLTemp.rhoLTE.histedges[i_ind, j_ind]
                supp = learningOutput[0].rhoLTemp.rhoLTE.supp[i_ind, j_ind]
                
                RhoMat[i_ind, j_ind, :] = evaluate_rhoLT(hist, histedges, supp, rp)
                RhoMat[j_ind, i_ind, :] = RhoMat[i_ind, j_ind, :]
        
        gravity_terms = {}
        gravity_terms['PhiMat'] = PhiMat
        gravity_terms['RhoMat'] = RhoMat
        gravity_terms['rp'] = rp
        
    else:
        raise Exception('It only calculates gravity terms for 3 different methods!')
    
    return gravity_terms
