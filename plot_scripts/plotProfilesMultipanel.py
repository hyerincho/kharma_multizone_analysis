from plotProfiles import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os
from astropy import units as u
from astropy import constants as const

G = const.G
c = const.c

def rg2kpc(r,M=6.5e9*u.Msun):
    rg = G*M/c**2
    return (r*rg).to('kpc').value

def kpc2rg(R,M=6.5e9*u.Msun):
    rg = G*M/c**2
    return (R*u.kpc/rg).to('')

#variableToLatex = {}
#variableToLatex['Mdot'] = '$\dot{M}$ [arbitrary units]'
#variableToLatex['rho'] = r'$\rho$ [arbitrary units]'
#variableToLatex['u^r'] = '$u^r$'
#variableToLatex['T'] = '$\Theta = k\,T/(\mu\, c^2)$'

def plotProfilesMultipanel(listOfPickles, listOfLabels=None, listOfColors=None, listOfLinestyles=None, output=None, quantities=['Mdot', 'rho', 'T', 'u^r'], figsize=(12,8), rescale=False, rescaleRadius=10, rescaleValue=1, \
    fontsize=18, xlim=(2,4e9), ylim=(1e-2,10), show_init=True, show_gizmo=True, cta=10, boxcar_factor=0, average_factor=2, show_rscale=False, show_mdotinout=False, show_pc=False, set_beta_ylim=False):

    row = 2
    fig, axarr = plt.subplots(row, len(quantities)//row, figsize=figsize, sharex=True)
    ax1d = axarr.reshape(-1)
    if listOfLabels is None:
        listOfLabels = [None]*len(listOfPickles)
    if listOfColors is None:
        listOfColors = [None]*len(listOfPickles)
    if listOfLinestyles is None:
        listOfLinestyles = [None]*len(listOfPickles)

    if show_pc:
        r_bondi = np.logspace(np.log10(max(2,xlim[0])), np.log10(xlim[1]), 50)
        for ax in axarr[0,:]:
            ax.plot(r_bondi, np.zeros(len(r_bondi)))
            secax = ax.secondary_xaxis('top', functions=(rg2kpc, kpc2rg))
            secax.set_xlabel(r'$R_{\rm M87}$ [kpc]',fontsize=fontsize)
            secax.tick_params(axis='x', labelsize=fontsize-4)
    for col in range(ax1d.shape[0]):
        do_rescale = ('Mdot' in quantities[col] and rescaleValue != 1)
        ax = ax1d[col]
        if col == 0: label_list = listOfLabels
        else: label_list = ['__nolegend__']*len(listOfPickles)
        plotProfiles(listOfPickles, quantities[col], formatting=False, finish=False, label_list=label_list, color_list=listOfColors, linestyle_list=listOfLinestyles, fig_ax=(fig, ax), \
                flip_sign=(quantities[col] in ['u^r']), show_init=show_init, show_gizmo=show_gizmo, show_bondi=1, show_rscale=show_rscale, show_mdotinout=show_mdotinout, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                rescale=do_rescale, rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor, average_factor=average_factor) #
        if quantities[col] == 'eta': ax.axhline(0.01,color='m',lw=3,alpha=0.2) #ax.axhspan(0.01,0.02,color='m',alpha=0.2) # horizontal line to show 2%
        if quantities[col] == 'beta': ax.legend(loc='best', frameon=False, fontsize=fontsize-4)
        if show_mdotinout and quantities[col] == "Mdot":
            plotProfiles(listOfPickles, 'Mdot_in', formatting=False, finish=False, label_list=[r'$\dot{M}_{\rm in}$'], color_list=['b'], linestyle_list=listOfLinestyles, fig_ax=(fig, ax), \
                    flip_sign=(quantities[col] in ['u^r']), show_init=0, show_gizmo=show_gizmo, show_bondi=1, show_rscale=show_rscale, show_mdotinout=show_mdotinout, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                    rescale=do_rescale, rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor) #
            plotProfiles(listOfPickles, 'Mdot_out', formatting=False, finish=False, label_list=[r'$\dot{M}_{\rm out}$'], color_list=['r'], linestyle_list=listOfLinestyles, fig_ax=(fig, ax), \
                    flip_sign=(quantities[col] in ['u^r']), show_init=0, show_gizmo=show_gizmo, show_bondi=1, show_rscale=show_rscale, show_mdotinout=show_mdotinout, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                    rescale=do_rescale, rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor) #
            ax.legend(loc='best', frameon=False, fontsize=fontsize-4)
        ax.tick_params(axis='both', labelsize=fontsize-4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if do_rescale:
            ax.set_title(variableToLabel(quantities[col]).replace('arb. units', r'$\dot{M}_B$'))
        else: 
            if show_pc:
                ax.set_ylabel(variableToLabel(quantities[col]), fontsize=fontsize)
            else:
                ax.set_title(variableToLabel(quantities[col]), fontsize=fontsize)
        ax.set_xlim(xlim)
        if quantities[col] == 'Mdot':
            ax.set_ylim(ylim)
        elif quantities[col] == 'eta':
            ax.set_ylim([1e-4,1])
        elif 'Omega' in quantities[col]:
            ax.set_ylim([1e-3,10])
        elif quantities[col] == 'phib':
            ax.set_ylim([10,1e6])
            ax.legend(loc='best',frameon=False,fontsize=fontsize-4,handlelength=2)
        if set_beta_ylim and quantities[col] == 'beta':
            ax.set_ylim([1e-2,10])
    
    for col in range(axarr.shape[1]): axarr[-1,col].set_xlabel('$r \ [r_g]$', fontsize=fontsize)


    #for run_index in range(len(listOfPickles)):
    #    ax1d[lgd_ax].plot([], [], color=listOfColors[run_index], lw=2, label=listOfLabels[run_index], ls=listOfLinestyles[run_index])
    if len(quantities) > 4: handlelength=2
    else: handlelength=5
    for i in range(2): ax1d[i].legend(loc='best', frameon=False, fontsize=fontsize-4, handlelength=handlelength)

    fig.tight_layout()
    if show_pc:
        plt.subplots_adjust(wspace=0.23,hspace=0.01)
    else:
        plt.subplots_adjust(wspace=0.2,hspace=0.2)
    if output is None:
        fig.show()
    else:
        fig.savefig(output,bbox_inches='tight')
        plt.close(fig)

def plotHydro():
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all.pkl', 'production_runs/gizmo_extg_1e8_profiles_all.pkl']]
    listOfLabels = [r'Bondi $\S \, 3.1$', r'Ext.Grav. $\S 3.2$', 'Bondi+Ext.Profiles', 'Bondi+Rot.']
    listOfColors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green'] #'k', 
    listOfLinestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    xlim = (2, 4e9)
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, xlim=xlim, show_pc=True, output='../plots/combined_profiles.pdf')

def plotMHDtest():
    listOfPickles = ['../data_products/'+dirname for dirname in ['082423_2d_onezone_profiles_all2.pkl','082823_2d_n4_longtout_profiles_all2.pkl','production_runs/072823_beta01_onezone_profiles_all2.pkl','082423_n4_profiles_all2.pkl']]# '083123_ozrst_onezone_profiles_all2.pkl', 
    listOfLabels = [r'$\S \,4.1$ weak 1-zone', r'$\S \,4.1$ weak 4-zone',r'$\S \,4.2$ strong 1-zone', r'$\S \,4.2$ strong 4-zone']
    listOfColors = ['tab:red','tab:orange','k', 'tab:blue', 'tab:orange', 'tab:green']
    listOfLinestyles = ['solid','dashed', 'solid', 'solid','dashdot', 'dotted']
    quantities = ['Mdot', 'rho', 'T', 'u^r', 'beta', 'eta'] #, 'abs_Omega', 'T'
    xlim = (2, 4e4)
    ylim = (1e-4,2)
    cta= 0 #10
    rescaleValue = bondi.get_quantity_for_rarr([1e3], 'Mdot', rs=16)[0]
    boxcar_factor = 0 #4
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, \
            xlim=xlim, ylim=ylim, show_init=False, show_gizmo=False, show_rscale=True, quantities=quantities, rescaleValue=rescaleValue, cta=cta, \
            boxcar_factor=boxcar_factor, average_factor=2, \
            output='../plots/combined_profiles.pdf')
def plotMHD():
    #listOfPickles = ['../data_products/'+dirname for dirname in ['082423_n8_profiles_all2.pkl']]
    listOfPickles = ['../data_products/'+dirname for dirname in ['production_runs/072823_beta01_128_profiles_all2.pkl']]
    listOfLabels = ['__nolegend__']
    listOfColors = ['k', 'tab:blue', 'tab:orange', 'tab:green']
    listOfLinestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    quantities = ['Mdot', 'rho', 'T', 'phib', 'beta', 'eta'] #, 'Omega'
    figsize=(14,8)
    xlim = (2, 2e8)
    ylim = (1e-4,2)
    cta= 0
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    boxcar_factor = 4
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, \
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale='rho_phib', show_mdotinout=True, figsize=figsize, quantities=quantities, rescaleValue=rescaleValue, cta=cta, boxcar_factor=boxcar_factor,\
            set_beta_ylim=1,average_factor=2,output='../plots/combined_profiles.pdf')

if __name__ == '__main__':
    #plotHydro()
    #plotMHDtest()
    plotMHD()
