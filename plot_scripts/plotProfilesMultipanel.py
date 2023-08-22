from plotProfiles import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_settings import *
import pickle
import os

variableToLatex = {}
variableToLatex['Mdot'] = '$\dot{M}$ [arbitrary units]'
variableToLatex['rho'] = r'$\rho$ [arbitrary units]'
variableToLatex['u^r'] = '$u^r$'
variableToLatex['T'] = '$\Theta = k\,T/(\mu\, c^2)$'

def plotProfilesMultipanel(listOfPickles, listOfLabels=None, listOfColors=None, listOfLinestyles=None, output=None, quantities=['Mdot', 'rho', 'T', 'u^r'], figsize=(10,8), rescale=False, rescaleRadius=10, rescaleValue=1, \
    fontsize=18, xlim=(2,4e9)):

    row = 2
    fig, axarr = plt.subplots(row, len(quantities)//row, figsize=figsize, sharex=True)
    ax1d = axarr.reshape(-1)
    if listOfLabels is None:
        listOfLabels = [None]*len(listOfPickles)
    if listOfColors is None:
        listOfColors = [None]*len(listOfPickles)
    if listOfLinestyles is None:
        listOfLinestyles = [None]*len(listOfPickles)

    for col in range(ax1d.shape[0]):
        ax = ax1d[col]
        plotProfiles(listOfPickles, quantities[col], formatting=False, finish=False, label_list=listOfLabels, color_list=listOfColors, linestyle_list=listOfLinestyles, fig_ax=(fig, ax1d[col]), \
        flip_sign=(quantities[col] in ['u^r']), show_init=1, show_gizmo=1, show_bondi=1, cycles_to_average=10, show_divisions=0, show_rb=1)#, rescale=(quantities[col] in ['Mdot', 'rho']))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(variableToLatex[quantities[col]], fontsize=fontsize)
        ax.set_xlim(xlim)
        if quantities[col] == 'Mdot':
            ax.set_ylim([1e-2,10])
    
    for col in range(axarr.shape[1]): axarr[-1,col].set_xlabel('$r \ [r_g]$', fontsize=fontsize)

    for run_index in range(len(listOfPickles)):
        ax1d[0].plot([], [], color=listOfColors[run_index], lw=2, label=listOfLabels[run_index], ls=listOfLinestyles[run_index])
        ax1d[0].legend(loc='best', frameon=False, fontsize=fontsize-4)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    if output is None:
        fig.show()
    else:
        fig.savefig(output,bbox_inches='tight')
        plt.close(fig)

def plotHydro():
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all.pkl', 'production_runs/gizmo_extg_1e8_profiles_all.pkl']]
    listOfLabels = [r'Bondi $\S \, 3.1$', r'Ext.Grav. $\S 3.2$', 'Bondi+Ext.Profiles', 'Bondi+Rot.']
    listOfColors = ['k', 'tab:blue', 'tab:orange', 'tab:green']
    listOfLinestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, output='../plots/combined_profiles.pdf')

if __name__ == '__main__':
    plotHydro()
