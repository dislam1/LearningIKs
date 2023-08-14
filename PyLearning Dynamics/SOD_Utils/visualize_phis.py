import numpy as np
import matplotlib.pyplot as plt

from SOD_Utils.plot_joint_distribution import plot_joint_distribution
from SOD_Utils.plot_one_type_of_phis import plot_one_type_of_phis

def visualizephis(learningOutput, sys_info, obs_info, learn_info, plot_info):
    # ... (code to handle screen size and other variables)

   
    # Prepare different phi's
    n_trials = len(learningOutput)
    phihat = [None] * n_trials
    phihatsmooth = [None] * n_trials
    basis = [None] * n_trials
    phi = sys_info['phiE']

    if phi is not None:
        basis_info = learn_info['Ebasis_info']
        for ind in range(n_trials):
            phihat[ind] = learningOutput[ind]['Estimator']['phiEhat']
            phihatsmooth[ind] = learningOutput[ind]['Estimator']['phiEhatsmooth']
            basis[ind] = learningOutput[ind]['Estimator']['Ebasis']

        # Prepare the window to plot the true and learned interactions
        phi_fig, ax = plt.subplots(figsize=(12, 9))
        plot_info['phi_type'] = 'energy'
        plot_one_type_of_phis(phi_fig, phi, phihat, phihatsmooth, learningOutput, basis_info, sys_info, obs_info, plot_info)
        if 'plot_name' in plot_info:
            plt.savefig(f"{plot_info['plot_name']}_phiE")

    if sys_info['ode_order'] == 2:
        phi = sys_info['phiA']
    else:
        phi = None

    if phi is not None:
        basis_info = learn_info['Abasis_info']
        for ind in range(n_trials):
            phihat[ind] = learningOutput[ind]['Estimator']['phiAhat']
            phihatsmooth[ind] = learningOutput[ind]['Estimator']['phiAhatsmooth']
            basis[ind] = learningOutput[ind]['Estimator']['Abasis']

        # Prepare the window to plot the true and learned interactions
        phi_fig, ax = plt.subplots(figsize=(12, 9))
        plot_info['phi_type'] = 'alignment'
        plot_one_type_of_phis(phi_fig, phi, phihat, phihatsmooth, learningOutput, basis_info, sys_info, obs_info, plot_info)
        plt.savefig(f"{plot_info['plot_name']}_phiA")

        # Prepare the window to plot the joint distribution
        rhoLTA_fig, ax = plt.subplots(figsize=(12, 9))
        plot_joint_distribution(rhoLTA_fig, obs_info['rhoLT']['rhoLTA'], learningOutput[0]['rhoLTemp']['rhoLTA'], sys_info['K'], plot_info)
        plt.savefig(f"{plot_info['plot_name']}_rhoLTA")

    if sys_info['ode_order'] == 2 and sys_info['has_xi']:
        phi = sys_info['phiXi']
    else:
        phi = None

    if phi is not None:
        basis_info = learn_info['Xibasis_info']
        for ind in range(n_trials):
            phihat[ind] = learningOutput[ind]['Estimator']['phiXihat']
            phihatsmooth[ind] = learningOutput[ind]['Estimator']['phiXihatsmooth']
            basis[ind] = learningOutput[ind]['Estimator']['Xibasis']

        # Prepare the window to plot the true and learned interactions
        phi_fig, ax = plt.subplots(figsize=(12, 9))
        plot_info['phi_type'] = 'xi'
        plot_one_type_of_phis(phi_fig, phi, phihat, phihatsmooth, learningOutput, basis_info, sys_info, obs_info, plot_info)
        plt.savefig(f"{plot_info['plot_name']}_phiXi")

        # Prepare the window to plot the joint distribution
        rhoLTXi_fig, ax = plt.subplots(figsize=(12, 9))
        plot_joint_distribution(rhoLTXi_fig, obs_info['rhoLT']['rhoLTXi'], learningOutput[0]['rhoLTemp']['rhoLTXi'], sys_info['K'], plot_info)
        plt.savefig(f"{plot_info['plot_name']}_rhoLTXi")
    if sys_info['debug_mode']:
        plt.show()
    else:
        plt.ion()
