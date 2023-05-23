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
variableToLatex['T'] = '$T \ [m_e c^2]$'

def plotProfilesMultipanel(listOfPickles, listOfLabels=None, listOfColors=None, listOfLinestyles=None, output=None, quantities=['Mdot', 'rho', 'u^r', 'T'], figsize=(12,3.8), rescale=False, rescaleRadius=10, rescaleValue=1, \
	fontsize=13, xlim=(2,2e7)):

	fig, axarr = plt.subplots(1, len(quantities), figsize=figsize)
	if listOfLabels is None:
		listOfLabels = [None]*len(listOfPickles)
	if listOfColors is None:
		listOfColors = [None]*len(listOfPickles)
	if listOfLinestyles is None:
		listOfLinestyles = [None]*len(listOfPickles)

	for col in range(axarr.shape[0]):
		ax = axarr[col]
		plotProfiles(listOfPickles, quantities[col], formatting=False, finish=False, final_only=True, label_list=listOfLabels, color_list=listOfColors, linestyle_list=listOfLinestyles, fig_ax=(fig, axarr[col]), \
		flip_sign=(quantities[col] in ['u^r']), rescale=(quantities[col] in ['Mdot', 'rho']), rescaleRadius=rescaleRadius, rescaleValue=rescaleValue)
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlabel('$r \ [r_g]$', fontsize=fontsize)
		ax.set_ylabel(variableToLatex[quantities[col]], fontsize=fontsize)
		ax.set_xlim(xlim)

	for run_index in range(len(listOfPickles)):
		axarr[0].plot([], [], color=listOfColors[run_index], lw=2, label=listOfLabels[run_index], ls=listOfLinestyles[run_index])
		axarr[0].legend(loc='best', frameon=False, fontsize=fontsize-4)

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)
		plt.close(fig)

if __name__ == '__main__':
	listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_022823_bondi_new_coord_noffp_profiles.pkl', 'bondi_multizone_040223_gizmo_ext_g_64^3_lin_profiles.pkl', 'bondi_multizone_032723_gizmo_no_ext_g_64^3_lin_profiles.pkl', 'bondi_multizone_040423_bondi_64^3_rot_profiles.pkl']]
	listOfLabels = ['Bondi', 'Bondi+Ext.Profiles\n+Ext.Grav.', 'Bondi+Ext.Profiles', 'Bondi+Rot.']
	listOfColors = ['k', 'tab:blue', 'tab:orange', 'tab:green']
	listOfLinestyles = ['solid', 'dashed', 'dashdot', 'dotted']
	plotProfilesMultipanel(listOfPickles, listOfLabels=listOfLabels, listOfColors=listOfColors, listOfLinestyles=listOfLinestyles, output='../plots/combined_profiles.pdf')
