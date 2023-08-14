import numpy as np
import matplotlib.pyplot as plt
from SOD_Utils.plot_joint_distribution import plot_joint_distribution
from SOD_Utils.plot_phiA_and_phiXi_together import plot_phiA_and_phiXi_together

def visualizephisforPT(learningOutput, sys_info, obs_info, plot_info):
    # Screen size issue
    if 'scrsz' in plot_info:
        scrsz = plot_info['scrsz']
    else:
        scrsz = plt.gcf().get_size_inches() * plt.gcf().dpi

    # Plot phiA and phiXi together
    phi_fig, ax = plt.subplots(figsize=(scrsz[0]*3/4, scrsz[1]*3/4))
    plot_phiA_and_phiXi_together(phi_fig, learningOutput, sys_info, obs_info, plot_info)
    plt.savefig(f"{plot_info['plot_name']}_phiA_n_phiXi")

    # Prepare the window to plot the joint distribution: rhoLTA
    rhoLTA_fig, ax = plt.subplots(figsize=(scrsz[0]*3/4, scrsz[1]*3/4))
    plot_info['phi_type'] = 'alignment'
    plot_joint_distribution(rhoLTA_fig, obs_info['rhoLT']['rhoLTA'], learningOutput[0]['rhoLTemp']['rhoLTA'], sys_info['K'], plot_info)
    plt.savefig(f"{plot_info['plot_name']}_rhoLTA")

    # Prepare the window to plot the joint distribution: rhoLTXi
    rhoLTXi_fig, ax = plt.subplots(figsize=(scrsz[0]*3/4, scrsz[1]*3/4))
    plot_info['phi_type'] = 'xi'
    plot_joint_distribution(rhoLTXi_fig, obs_info['rhoLT']['rhoLTXi'], learningOutput[0]['rhoLTemp']['rhoLTXi'], sys_info['K'], plot_info)
    plt.savefig(f"{plot_info['plot_name']}_rhoLTXi")

    plt.show()
