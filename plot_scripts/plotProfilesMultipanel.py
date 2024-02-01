from plotProfiles import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os
import string
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
    fontsize=18, xlim=(2,4e9), ylim=(1e-2,10), show_init=False, show_gizmo=True, cta=10, boxcar_factor=0, average_factor=2, show_rscale=False, show_mdotinout=None, show_pc=False, show_legend=None, show_eta_axhline=True, show_band_unconverged=False, set_beta_ylim=False,eta_norm_Bondi=False, row= None, tmax_list=None, lw= 2):
    
    matplotlib_settings()
 
    if len(quantities) <=2: row=1
    elif row is None: row = 2
    fig, axarr = plt.subplots(row, len(quantities)//row, figsize=figsize, sharex=True)
    ax1d = axarr.reshape(-1)
    if listOfLabels is None:
        listOfLabels = [None]*len(listOfPickles)
    if listOfColors is None:
        listOfColors = [None]*len(listOfPickles)
    if listOfLinestyles is None:
        listOfLinestyles = ['solid']*len(listOfPickles)

    if show_pc:
        r_bondi = np.logspace(np.log10(max(2,xlim[0])), np.log10(xlim[1]), 50)
        if len(quantities) <= 2 : axarr_temp = axarr[np.newaxis,:]
        else: axarr_temp = axarr
        for ax in axarr_temp[0,:]:
            ax.plot(r_bondi, np.zeros(len(r_bondi)))
            secax = ax.secondary_xaxis('top', functions=(rg2kpc, kpc2rg))
            secax.set_xlabel(r'$R_{\rm M87}$ [kpc]',fontsize=fontsize, labelpad=7)
            secax.tick_params(axis='x', labelsize=fontsize-4)
    for col in range(ax1d.shape[0]):
        do_rescale = ('Mdot' in quantities[col] and rescaleValue != 1)
        ax = ax1d[col]
        if col == 0 or (show_legend is not None and col in show_legend[1:]): label_list = listOfLabels
        else: label_list = ['__nolegend__']*len(listOfPickles)
        plotProfiles(listOfPickles, quantities[col], formatting=False, finish=False, label_list=label_list, color_list=listOfColors, linestyle_list=listOfLinestyles, fig_ax=(fig, ax), \
                flip_sign=(quantities[col] in ['u^r']), show_init=show_init, show_gizmo=show_gizmo, show_bondi=1, show_rscale=show_rscale, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                rescale=do_rescale, rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor, average_factor=average_factor, eta_norm_Bondi=eta_norm_Bondi, lw=lw, tmax_list=tmax_list) #
        if show_eta_axhline and (quantities[col] == 'eta' and not eta_norm_Bondi): ax.axhline(0.01,color='m',lw=3,alpha=0.5) #ax.axhspan(0.01,0.02,color='m',alpha=0.2) # horizontal line to show 2%
        if quantities[col] == 'beta': ax.legend(loc='best', frameon=False, fontsize=fontsize-4)
        # show Mdot in and Mdot out
        if show_mdotinout is not None and quantities[col] == "Mdot":
            if show_mdotinout == True: show_mdotinout = slice(None)
            plotProfiles(listOfPickles[show_mdotinout], 'Mdot_in', formatting=False, finish=False, label_list=[r'MHD $\dot{M}_{\rm in}$']*len(listOfPickles), color_list=['b']*len(listOfPickles), linestyle_list=listOfLinestyles, fig_ax=(fig, ax), \
                    flip_sign=(quantities[col] in ['u^r']), show_init=0, show_gizmo=show_gizmo, show_rscale=show_rscale, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                    rescale=do_rescale, rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor,lw=1.5, tmax_list=tmax_list[show_mdotinout]) #
            plotProfiles(listOfPickles[show_mdotinout], 'Mdot_out', formatting=False, finish=False, label_list=[r'MHD $\dot{M}_{\rm out}$']*len(listOfPickles), color_list=['deepskyblue']*len(listOfPickles), linestyle_list=listOfLinestyles, fig_ax=(fig, ax), \
                    flip_sign=(quantities[col] in ['u^r']), show_init=0, show_gizmo=show_gizmo, show_rscale=show_rscale, cycles_to_average=cta, show_divisions=0, show_rb=1, \
                    rescale=do_rescale, rescale_value=rescaleValue, num_time_chunk=1, boxcar_factor=boxcar_factor, lw=2, tmax_list=tmax_list[show_mdotinout]) #
            ax.legend(loc='center left', frameon=False, fontsize=fontsize-4)
        ax.tick_params(axis='both', labelsize=fontsize-4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        label = variableToLabel(quantities[col])
        if do_rescale:
            label = label.replace('arb. units', r'$\dot{M}_B$')
        if eta_norm_Bondi and quantities[col]=='eta':
            label = r'$\overline{\dot{M}-\dot{E}}/\dot{M}_B$'
        if show_pc:
            ax.set_ylabel(label, fontsize=fontsize)
        else:
            ax.set_title(label, fontsize=fontsize)
        ax.set_xlim(xlim)
        if quantities[col] == 'Mdot':
            ax.set_ylim(ylim)
        elif quantities[col] == 'eta' and not eta_norm_Bondi:
            ax.set_ylim([2e-6,1e-1])
        elif 'Omega' in quantities[col]:
            ax.set_ylim([1e-3,10])
        elif quantities[col] == 'phib':
            ax.set_ylim([10,1e6])
            ax.legend(loc='best',frameon=False,fontsize=fontsize-4,handlelength=2)
        if set_beta_ylim and quantities[col] == 'beta':
            ax.set_ylim([1e-2,10])
    
    if len(quantities)<=2: 
        for col in range(axarr.shape[0]): axarr[col].set_xlabel('$r \ [r_g]$', fontsize=fontsize)
    else: 
        for col in range(axarr.shape[1]): axarr[-1,col].set_xlabel('$r \ [r_g]$', fontsize=fontsize)
    if show_band_unconverged:
        for ax in ax1d: ax.axvspan(1e6,xlim[1],facecolor='gray',linewidth=0.0,alpha=0.5)

    #for run_index in range(len(listOfPickles)):
    #    ax1d[lgd_ax].plot([], [], color=listOfColors[run_index], lw=2, label=listOfLabels[run_index], ls=listOfLinestyles[run_index])
    #if len(quantities) > 4:
    handlelength=2
    #else: handlelength=5
    if show_legend is None:
        for i in range(2): ax1d[i].legend(loc='best', frameon=False, fontsize=fontsize-4, handlelength=handlelength)
    else: 
        for i in show_legend:
            #ax1d[i].legend(loc='best', frameon=False, fontsize=fontsize-4, handlelength=handlelength)
            ax1d[i].legend(bbox_to_anchor=(0.4,1.1), ncol=4, frameon=False, fontsize=fontsize-4, handlelength=handlelength)

    # panel numbers
    for i in range(len(ax1d)): ax1d[i].text(0.02, 0.92, '('+string.ascii_lowercase[i]+')',transform=ax1d[i].transAxes, fontsize=fontsize-2)#, bbox=dict(facecolor='w', edgecolor='k', pad=5.0))

    if 1:
        # remove first ytick on the last row in the panels
        for a in ax1d:
            plt.setp(a.get_yticklabels()[0], visible=False)
            plt.setp(a.get_yticklabels()[-1], visible=False)

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
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all2.pkl', 'production_runs/gizmo_extg_1e8_profiles_all.pkl']]
    listOfLabels = [r'Bondi $\S \, 3.1$', r'Ext.Grav. $\S 3.2$', 'Bondi+Ext.Profiles', 'Bondi+Rot.']
    listOfColors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green'] #'k', 
    listOfLinestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    xlim = (2, 4e9)
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, xlim=xlim, show_pc=True, output='../plots/combined_profiles.pdf')

def plotMHDtest():
    listOfPickles = ['../data_products/'+dirname for dirname in ['082423_2d_onezone_profiles_all2.pkl','082823_2d_n4_longtout_profiles_all2.pkl','production_runs/072823_beta01_onezone_profiles_all2.pkl','101623_n4_cap_profiles_all2.pkl']]# '083123_ozrst_onezone_profiles_all2.pkl', '082423_n4_profiles_all2.pkl'
    listOfLabels = [r'$\S \,4.1$ weak 1-zone', r'$\S \,4.1$ weak 4-zone',r'$\S \,4.2$ strong 1-zone', r'$\S \,4.2$ strong 4-zone']
    listOfColors = ['tab:red','tab:orange','k', 'tab:blue', 'tab:orange', 'tab:green']
    listOfLinestyles = ['solid','dashed', 'solid', 'solid','dashdot', 'dotted']
    quantities = ['Mdot', 'rho', 'T', 'u^r', 'beta', 'eta'] #, 'abs_Omega', 'T'
    xlim = (2, 4e4)
    ylim = (1e-4,2)
    cta= 0 #10
    rescaleValue = bondi.get_quantity_for_rarr([1e3], 'Mdot', rs=16)[0]
    boxcar_factor = 0 #4
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, tmax_list=[None, None, None, 30],\
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
            set_beta_ylim=1,average_factor=np.sqrt(2),output='../plots/combined_profiles.pdf')

def plotLetterPrims():
    # For the letter. 4 panel showing rho, T, phib, beta
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all2.pkl', 'production_runs/072823_beta01_128_profiles_all2.pkl']]
    listOfLabels = ['Bondi HD', 'Bondi MHD']
    listOfColors = ['tab:red', 'k', 'tab:blue', 'tab:orange', 'tab:green'] #'k', 
    listOfLinestyles = ['solid', 'solid', 'dashdot', 'dotted']
    quantities = ['rho', 'T', 'beta', 'phib'] #, 'Omega'
    xlim = (2, 2e8)
    cta = 0
    boxcar_factor = 4
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, show_pc=True, \
            xlim=xlim, show_gizmo=False, show_rscale='rho_T_phib', quantities=quantities, cta=cta, boxcar_factor=boxcar_factor,\
            set_beta_ylim=1,average_factor=2,output='../plots/combined_profiles.pdf')

def plotLetterCons():
    # For the letter. 2 panel showing the conserved variables Mdot, eta
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all2.pkl', 'production_runs/072823_beta01_128_profiles_all2.pkl']]
    listOfLabels = ['HD (net)', 'MHD (net)']
    listOfColors = ['tab:red', 'k', 'tab:blue', 'tab:orange', 'tab:green'] #'k', 
    listOfLinestyles = ['solid', 'solid', 'dashdot', 'dotted']
    quantities = ['Mdot', 'eta'] 
    xlim = (2, 2e8)
    ylim = (1e-4,5)
    figsize=(14,6)
    cta = 0
    boxcar_factor = 4
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, show_pc=True, \
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale='rho_phib', show_mdotinout=slice(1,2), quantities=quantities, cta=cta, boxcar_factor=boxcar_factor,\
            set_beta_ylim=1,average_factor=2,figsize=figsize, rescaleValue=rescaleValue,output='../plots/letter_cons_profiles.pdf')

def plotLetter():
    # For the letter. 6 panel showing rho, T, phib, beta, Mdot, eta
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all2.pkl']]#, 'production_runs/072823_beta01_128_profiles_all2.pkl']]
    listOfLabels = ['Bondi HD', 'Bondi MHD']
    listOfColors = ['tab:red', 'k', 'tab:blue', 'tab:orange', 'tab:green'] #'k', 
    listOfLinestyles = ['solid', 'solid', 'dashdot', 'dotted']
    quantities = ['rho', 'T', 'Mdot', 'eta', 'beta', 'phib']
    xlim = (2, 2e8)
    ylim = (3e-4,5)
    figsize=(14,12)
    cta = 0
    boxcar_factor = 4
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, show_pc=True,show_eta_axhline=1, show_init=0, \
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale='rho_T_phib', show_mdotinout=slice(1,2), quantities=quantities, cta=cta, boxcar_factor=boxcar_factor,\
            set_beta_ylim=1,average_factor=2, row=3, figsize=figsize, rescaleValue=rescaleValue, lw=3, tmax_list=[227,9], output='../plots/letter_profiles.pdf')# tmax_list=[170,9] tmax_list=[600,9] 107,9

def plotN8ResComp():
    listOfPickles = ['../data_products/'+dirname for dirname in ['production_runs/072823_beta01_128_profiles_all2.pkl', '091423_n8_48_profiles_all2.pkl', '082423_n8_profiles_all2.pkl', '091523_n8_96_fmks_profiles_all2.pkl', '091723_n8_beta10_profiles_all2.pkl', '102323_n8_constrho_profiles_all2.pkl', '102423_n8_bondi_profiles_all2.pkl']] #, '091423_n8_96_profiles_all2.pkl'
    listOfLabels = [r'$128^3$, fmks', r'(i) $48^3$', r'(ii) $64^3$', r'(iii) $96^3$, fmks', r'(iv) $64^3$, $\beta$10', r'(v) $64^3$, const', r'(vi) $64^3$, bondi'] #r'$96^3$', 
    #listOfColors = ['k', 'tab:green','tab:blue', 'm', 'y', 'b', 'deepskyblue'] #,'tab:orange'
    cmap = plt.get_cmap('gist_rainbow')
    listOfColors = ['k'] + list(cmap(np.linspace(0.05,0.9,len(listOfPickles)-1)))
    listOfLinestyles = None
    quantities = ['rho', 'T','Mdot', 'eta', 'phib', 'beta']
    figsize=(15,8)
    xlim = (2, 2e8)
    ylim = (1e-4,2)
    cta= 0
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    boxcar_factor = 4
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, \
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale=False, show_legend=[0], figsize=figsize, quantities=quantities, rescaleValue=rescaleValue, cta=cta, boxcar_factor=boxcar_factor,\
            set_beta_ylim=1,average_factor=2,eta_norm_Bondi=0,tmax_list=[9,9,9,9,9,8.7,12], output='../plots/letter_rescomp_profiles.pdf')

def plotProposal():
    # For the proposal. 6 panel showing rho, T, phib, beta, Mdot, eta
    listOfPickles = ['../data_products/'+dirname for dirname in ['082423_n8_profiles_all2.pkl', '101123_gizmo_mhd_3d_profiles_all2.pkl']] #'production_runs/072823_beta01_128_profiles_all2.pkl'
    listOfLabels = ['Bondi MHD', 'GIZMO MHD']
    listOfColors = ['k', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange'] #'k', 
    listOfLinestyles = ['solid', 'solid', 'dashdot', 'dotted']
    quantities = ['rho', 'T', 'eta', 'beta']
    figsize=(11,8)
    xlim = (2, 1e7)
    ylim = (3e-4,5)
    cta = 0
    boxcar_factor = 4
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, show_pc=True, figsize=figsize,\
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale='rho_T_phib', show_eta_axhline=False, quantities=quantities, cta=cta, boxcar_factor=boxcar_factor,\
            set_beta_ylim=1,average_factor=2, rescaleValue=rescaleValue, lw=3, tmax_list=[9,9], output='../plots/proposal_mhd_profiles.pdf')

def plotBABAM():
    # For BABAM conference lightning talk 2 panel showing rho, eta
    listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_030723_bondi_128^3_profiles_all2.pkl','production_runs/072823_beta01_128_profiles_all2.pkl']] #,'082423_n8_profiles_all2.pkl']]
    listOfLabels = ['HD','MHD']
    listOfColors = ['tab:red', 'k', 'tab:red', 'tab:green', 'tab:blue', 'tab:orange'] #'k', 
    listOfLinestyles = ['solid', 'solid', 'dashdot', 'dotted']
    quantities = ['rho', 'eta']
    figsize=(14,6)
    xlim = (2, 2e8)
    ylim = (3e-4,5)
    cta = 0
    boxcar_factor = 4
    rescaleValue = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
    plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, show_pc=True, show_eta_axhline=1, show_init=0, \
            xlim=xlim, ylim=ylim, show_gizmo=False, show_rscale='rho_T_phib', show_band_unconverged=1, quantities=quantities, cta=cta, boxcar_factor=boxcar_factor, figsize=figsize,\
            set_beta_ylim=1,average_factor=2, rescaleValue=rescaleValue, lw=3, tmax_list=[227,9], output='../plots/babam_profiles.png') # 1000


if __name__ == '__main__':
    #plotHydro()
    #plotMHDtest()
    #plotMHD()
    #plotLetterPrims()
    #plotLetterCons()
    #plotLetter()
    #plotN8ResComp()
    #plotProposal()
    plotBABAM()
