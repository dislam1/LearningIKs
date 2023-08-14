from SOD_Utils.relativeErrorInfluenceFunction import relativeErrorInfluenceFunction
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_prey import PS_1st_order_prey_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_prey import PS_1st_order_predator_on_prey
from SOD_Examples.Predators_Swarm.PS_1st_order_prey_on_predator import PS_1st_order_prey_on_predator
from SOD_Examples.Predators_Swarm.PS_1st_order_predator_on_predator import PS_1st_order_predator_on_predator

def computeL2rhoTErr(Estimator, sys_info, obs_info):
    L2rhoTErr = {}

    # Go through three kinds of interaction laws
    if sys_info['phiE']:
        L2rhoTErr['EErr'] = relativeErrorInfluenceFunction(Estimator['phiEhat'], sys_info['phiE'], sys_info, obs_info, Estimator['Ebasis'], 'energy')
        L2rhoTErr['EErrSmooth'] = relativeErrorInfluenceFunction(Estimator['phiEhatsmooth'], sys_info['phiE'], sys_info, obs_info, Estimator['Ebasis2'], 'energy')
    
    if sys_info['ode_order'] == 2 and sys_info['phiA']:
        L2rhoTErr['AErr'] = relativeErrorInfluenceFunction(Estimator['phiAhat'], sys_info['phiA'], sys_info, obs_info, Estimator['Abasis'], 'alignment')
        L2rhoTErr['AErrSmooth'] = relativeErrorInfluenceFunction(Estimator['phiAhatsmooth'], sys_info['phiA'], sys_info, obs_info, Estimator['Abasis2'], 'alignment')
    
    if sys_info['ode_order'] == 2 and sys_info['has_xi']:
        L2rhoTErr['XiErr'] = relativeErrorInfluenceFunction(Estimator['phiXihat'], sys_info['phiXi'], sys_info, obs_info, Estimator['Xibasis'], 'xi')
        L2rhoTErr['XiErrSmooth'] = relativeErrorInfluenceFunction(Estimator['phiXihatsmooth'], sys_info['phiXi'], sys_info, obs_info, Estimator['Xibasis2'], 'xi')

    return L2rhoTErr
