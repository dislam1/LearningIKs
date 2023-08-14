import numpy as np
import matplotlib.pyplot as plt
from SOD_Utils.downsampleHistCounts import downsampleHistCounts
from SOD_Utils.get_legend_name_for_rhos import get_legend_name_for_rhos
from SOD_Utils.get_exponent_scale import get_exponent_scale


def plot_rhos(fig_handle, ax, range_vals, rhoLT, rhoLTemp, k1, k2, sys_info, plot_info):
    #ax.set_fontsize(plot_info["tick_font_size"])
    #ax.set_fontname(plot_info["tick_font_name"])
    font = {'family': plot_info["tick_font_name"],
            'size': plot_info["tick_font_size"]}
    
    plt.rc('font', **font)

    the_color = ax.yaxis.label.get_color()
    num_RPH = 2 if rhoLT else 1
    rhoPlotHandles = [None] * num_RPH
    rhoPlotNames = [None] * num_RPH
    RPH_count = 0

    if len(rhoLT) > 0:
        RPH_count += 1
        edges = rhoLT["histedges"]
        edges_idxs = np.where((range_vals[0] <= edges) & (edges <= range_vals[1]))[0]
        end = edges_idxs.shape[0]
        if not np.isnan(rhoLT["hist"]).any():
            histdata1, edges = downsampleHistCounts(rhoLT["hist"][edges_idxs[:end-1]], edges[edges_idxs], int(np.sqrt(len(edges_idxs)))/2)
            centers = (edges[:-1] + edges[1:]) / 2
        else:
            centers = (edges[edges_idxs[:end-1]] + edges[edges_idxs[1:]]) / 2
            histdata1 = np.zeros(len(centers))

        centers = centers.ravel()
        #histdata1 = np.fliplr(histdata1)
        histHandle = ax.plot(centers, histdata1, 'k', linewidth=1)
        histHandle[0].set_color(the_color)
        rhoPlotHandles[RPH_count - 1] = ax.fill_between(centers, 0, histdata1, facecolor='lemonchiffon', edgecolor='none', alpha=0.3)
        rhoPlotNames[RPH_count - 1] = get_legend_name_for_rhos(sys_info, plot_info, 'rhoLT', k1, k2)
    else:
        histdata1 = []

    RPH_count += 1
    edges = rhoLTemp[0]["histedges"]
    edges_idxs = np.where((range_vals[0] <= edges) & (edges <= range_vals[1]))[0]
    if not np.isnan(rhoLTemp[0]["hist"]).any():
        histdata2, edges = downsampleHistCounts(rhoLTemp[0]["hist"][edges_idxs[:-1]], edges[edges_idxs], int(np.sqrt(len(edges_idxs)))/2)
        centers = (edges[:-1] + edges[1:]) / 2
    else:
        centers = (edges[edges_idxs[:-1]] + edges[edges_idxs[1:]]) / 2
        histdata2 = np.zeros(len(centers))
    centers = centers.ravel()
    #histdata2 = np.fliplr(histdata2)
    histHandle2 = ax.plot(centers, histdata2, 'k--', linewidth=1)
    histHandle2[0].set_color(the_color)
    rhoPlotHandles[RPH_count - 1] = ax.fill_between(centers, 0, histdata2, color=the_color, edgecolor='none', alpha=0.1)
    rhoPlotNames[RPH_count - 1] = get_legend_name_for_rhos(sys_info, plot_info, 'rhoLTemp', k1, k2)

    ax.axis('tight')
    if len(histdata1) > 0:
        tmpmax = plot_info["rhotscalingdownfactor"] * max(max(histdata1), max(histdata2))
    else:
        tmpmax = plot_info["rhotscalingdownfactor"] * max(histdata2)

    if tmpmax == 0:
        tmpmax = 2
    if np.isnan(tmpmax):
        tmpmax = 2

    #ax.set_ylim(0, tmpmax)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(get_exponent_scale(tmpmax),get_exponent_scale(tmpmax)))
    #ax.yaxis.get_major_formatter().set_powerlimits(get_exponent_scale(tmpmax))
    ax.yaxis.set_major_formatter('{:+.2g}'.format)

    return rhoPlotHandles, rhoPlotNames
