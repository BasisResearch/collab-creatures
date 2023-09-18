# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:51:17 2021


"""
import matplotlib.pyplot as plt


def setup_fig():
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 1
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['legend.title_fontsize'] = 7
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['xtick.minor.size'] = 1.5
    plt.rcParams['ytick.minor.size'] = 1.5
    plt.rcParams['xtick.minor.width'] = 0.4
    plt.rcParams['ytick.minor.width'] = 0.4
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['figure.figsize'] = [7.5 /2.54, 5/2.54] # 
    # plt.rcParams['figure.figsize'] = [6/2.54, 5/2.54] # 
    plt.rcParams['figure.figsize'] = [5/2.54, 4.5/2.54] #units are inches, numerator is in cm
    plt.rcParams['path.simplify'] = True
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['image.interpolation'] = None
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['patch.edgecolor'] = 'none'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.formatter.limits'] = (-3,3)


