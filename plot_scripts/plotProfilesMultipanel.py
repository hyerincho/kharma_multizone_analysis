from plotProfiles import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os

#variableToLatex = {}
#variableToLatex['Mdot'] = '$\dot{M}$ [arbitrary units]'
#variableToLatex['rho'] = r'$\rho$ [arbitrary units]'
#variableToLatex['u^r'] = '$u^r$'
#variableToLatex['T'] = '$\Theta = k\,T/(\mu\, c^2)$'

def plotProfilesMultipanel(listOfPickles, listOfLabels=None, listOfColors=None, listOfLinestyles=None, output=None, quantities=['Mdot', 'rho', 'T', 'u^r'], figsize=(10,8), rescale=False, rescaleRadius=10, rescaleValue=1, \
    fontsize=18, xlim=(2,4e9), ylim=(1e-2,10), show_gizmo=True, cta=10, boxcar_factor=0, show_rscale=False, lgd_ax=0):

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
                flip_sign=(quantities[col] in ['u^r']), show_init=1, show_gizmo=show_gizmo, show_bondi=1, show_rscale=show_rscale, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                rescale=(quantities[col] in ['Mdot']), rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor) #
        ax.set_xscale('log')
        ax.set_yscale('log')
        if 'Mdot' in quantities[col] and rescaleValue!=1:
            ax.set_title(variableToLabel(quantities[col]).replace('arb. units', r'$\dot{M}_B$'))
        else: ax.set_title(variableToLabel(quantities[col]), fontsize=fontsize)
        ax.set_xlim(xlim)
        if quantities[col] == 'Mdot':
            ax.set_ylim(ylim)
        elif quantities[col] == 'eta':
            ax.set_ylim([1e-4,1])
        elif quantities[col] == 'beta':
            ax.set_ylim([1e-2,10])
        elif quantities[col] == 'abs_Omega':
            ax.set_ylim([1e-3,10])
    
    for col in range(axarr.shape[1]): axarr[-1,col].set_xlabel('$r \ [r_g]$', fontsize=fontsize)

    for run_index in range(len(listOfPickles)):
        ax1d[lgd_ax].plot([], [], color=listOfColors[run_index], lw=2, label=listOfLabels[run_index], ls=listOfLinestyles[run_index])
        ax1d[lgd_ax].legend(loc='best', frameon=False, fontsize=fontsize-4)

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
    xlim = (2, 4e9)
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, xlim=xlim, output='../plots/combined_profiles.pdf')

def plotMHD():
    listOfPickles = ['../data_products/'+dirname for dirname in ['081723_rst64_longtin_save_profiles_all2.pkl']]
    listOfLabels = ['__nolegend__']
    listOfColors = ['k', 'tab:blue', 'tab:orange', 'tab:green']
    listOfLinestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    quantities = ['Mdot', 'rho', 'T', 'abs_Omega', 'eta', 'beta'] #, 'abs_Omega'
    xlim = (2, 2e8)
    ylim = (1e-4,2)
    cta= 0
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    boxcar_factor = 4
    lgd_ax=1
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, \
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale=True, quantities=quantities, rescaleValue=rescaleValue, cta=cta, boxcar_factor=boxcar_factor,\
            lgd_ax=1, output='../plots/combined_profiles.pdf')

if __name__ == '__main__':
    #plotHydro()
    plotMHD()
