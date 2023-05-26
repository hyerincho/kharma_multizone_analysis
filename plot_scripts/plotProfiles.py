import matplotlib.pyplot as plt
import numpy as np
import bondi_analytic as bondi
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os
import pdb

#TODO:  There should be a file in each folder that tells you how many zones there are, and other useful information.

def readQuantity(dictionary, quantity):

  invert = False
  if quantity == 'beta':
    #It's better to use inverse beta, if we have it.
    try:
      quantity_index = dictionary['quantities'].index('inv_beta')
      invert = True
    except:
      print("inv_beta doesn't exist, so we will stick with beta.")
      quantity_index = dictionary['quantities'].index(quantity)
    profiles = [profile_list[quantity_index] for profile_list in dictionary['profiles']]
  elif quantity == 'Omega':
    quantity_index_numerator = dictionary['quantities'].index('u^phi')
    quantity_index_denominator = dictionary['quantities'].index('u^t')
    profiles = [profile_list[quantity_index_numerator]/profile_list[quantity_index_denominator] for profile_list in dictionary['profiles']]
  else:
    quantity_index = dictionary['quantities'].index(quantity)
    profiles = [profile_list[quantity_index] for profile_list in dictionary['profiles']]
  return profiles, invert

def plotProfiles(listOfPickles, quantity, output=None, colormap='turbo', color_list=None, linestyle_list=None, figsize=(8,6), n_zones_list=[7], flip_sign=False, show_divisions=True,
  xlabel=None, ylabel=None, xlim=None, ylim=None, label_list=None, fig_ax=None, formatting=True, finish=True, rescale=False, rescale_radius=10, rescale_value=1, cycles_to_average=1, trim_zone=True):

  if isinstance(listOfPickles, str):
    listOfPickles = [listOfPickles]

  if label_list is None:
    label_list = [None]*len(listOfPickles)
  if color_list is None:
    color_list = [None]*len(listOfPickles)
  if linestyle_list is None:
    linestyle_list = [None]*len(listOfPickles)

  #Changes some defaults.
  matplotlib_settings()

  #If you want, provide your own figure and axis.  Good for multipanel plots.
  if fig_ax is None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
  else:
    fig, ax = fig_ax

  #Profiles are pre-computed.
  #See ../compute_scripts/computeProfiles.py for how this file is formatted.

  for run_index in range(len(listOfPickles)):
    with open(listOfPickles[run_index], 'rb') as openFile:
      D = pickle.load(openFile)

    radii = D['radii']
    #Formula that produces the zone number of a given run.  It would have been better to have this in some other file though.
    n_zones = n_zones_list[run_index]
    if n_zones > 1:
      zone_number_sequence = np.array([np.abs(np.abs(n_zones-1 - (i % (2*n_zones-2)))-(n_zones-1)) for i in range(len(radii))])
    else:
      zone_number_sequence = np.full(len(radii), 0)

    profiles, invert = readQuantity(D, quantity)
    n_profiles = len(profiles)
    if cycles_to_average > 0:
      r_plot = np.array([])
      values_plot = np.array([])
      #Only plot the very last iteration.  Recommended for comparing different initial conditions.
      rescalingFactor = 1.0
      if rescale:
        #Find rescaling factor.
        for r_set, profile_set in zip(radii,profiles):
          if (rescale_radius >= r_set[0]) & (rescale_radius <= r_set[-1]):
            rescalingFactor = rescale_value / np.interp(np.log10(rescale_radius), np.log10(r_set), profile_set)
      for zone in range(n_zones):
        matchingIndices = np.where(zone_number_sequence == zone)[0]
        if zone in [0,n_zones-1]:
          #Tricky edge case: there are half as many instances of the zones at the ends.
          cycles_to_average = int(np.ceil(cycles_to_average/2))
        indicesToAverage = matchingIndices[-np.min([len(matchingIndices),cycles_to_average]):]
        finalMatchingIndex = indicesToAverage[-1]
        n_radii = len(radii[finalMatchingIndex])
        plottable = np.zeros(n_radii)

        #Time average over some number of cycles.
        for indexToAverage in indicesToAverage:
          plottable += np.squeeze(profiles[indexToAverage])
        plottable /= len(indicesToAverage)

        #Flip the quantity upside-down, usually for inv_beta.
        if invert:
          plottable = 1.0 / plottable

        #Next, optionally mask out regions that likely have wall glitches by only taking the central half of the radii
        mask = np.full(n_radii, True, dtype=bool)
        if trim_zone:
          if zone > 0:
            mask[-int(n_radii/4):] = False
          if zone < n_zones - 1:
            mask[:int(n_radii/4)] = False

        r_plot = np.concatenate([r_plot,radii[finalMatchingIndex][mask]])
        values_plot = np.concatenate([values_plot,rescalingFactor*plottable[mask]*(-1)**int(flip_sign)])

      r_plot = np.squeeze(np.array(r_plot))
      values_plot = np.squeeze(np.array(values_plot))
      order = np.argsort(r_plot)
      ax.plot(r_plot[order], values_plot[order], color=color_list[run_index], ls=linestyle_list[run_index], lw=2)
    else:
      #Plot every iteration of a given run.  Recommended for examining a single run.  Rescaling not supported because you may be asked to interpolate outside the zone.
      for zone in range(n_profiles):
        valuesToColors = plt.cm.get_cmap(colormap)
        color = valuesToColors(zone/n_profiles)
        plottable = profiles[finalMatchingIndex]
        if invert:
          plottable = 1.0 / plottable
        ax.plot(radii[zone], plottable*(-1)**int(flip_sign), color=color, ls=linestyle_list[run_index], lw=2)

  #Formatting
  if formatting:
    if xlabel is None:
      xlabel = 'Radius [$r_g$]'
      ax.set_xlabel(xlabel)
    if ylabel is None:
      ylabel = variableToLabel(quantity)
      ax.set_ylabel(ylabel)  
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)
    for run_index in range(len(listOfPickles)):
      ax.plot([], [], color=color_list[run_index], lw=2, label=label_list[run_index], ls=linestyle_list[run_index])
      ax.legend(loc='best', frameon=False)
      fig.tight_layout()
    if show_divisions:
      divisions = []
      for zone in range(n_zones):
        divisions.append(radii[zone][-1])
        if (zone == n_zones-1) | (zone == n_zones-2):
          divisions.append(radii[zone][0])
      for div in divisions:
        ax.plot([div]*2, ax.get_ylim(), alpha=0.2, color='grey', lw=1)

  # Bondi analytic overplotting
  xlim = ax.get_xlim()
  r_bondi = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 50)
  analytic_sol = bondi.get_quantity_for_rarr(r_bondi,quantity,rs=16) # TODO: update this rs from pickle
  if analytic_sol is not None:
    ax.plot(r_bondi, analytic_sol,'k:',label='bondi analytic')

  #Either show or save.
  if finish:
    if output is None:
      fig.show()
    else:
      fig.savefig(output)
      plt.close(fig)

if __name__ == '__main__':

  # 2a) weak field test (n=4)
  listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4_profiles2.pkl', 'bondi_multizone_050423_bflux0_1e-8_2d_n4_profiles2.pkl']]
  listOfLabels = ['n=1', 'n=4']
  n_zones_list = [1, 4]
  plot_dir = '../plots/051523_weakfield'

  # 2b) strong field test
  '''
  listOfPickles = ['../data_products/'+dirname for dirname in \
      ['bondi_multizone_050123_onezone_bflux0_1e-4_64^3_profiles2.pkl', \
      'bondi_multizone_042723_bflux0_1e-4_32^3_profiles2.pkl', \
      'bondi_multizone_042723_bflux0_1e-4_64^3_profiles2.pkl', \
      'bondi_multizone_050123_bflux0_1e-4_96^3_profiles2.pkl', \
      'bondi_multizone_050523_bflux0_1e-4_128^3_n3_noshort_profiles2.pkl', \
      'bondi_multizone_050123_bflux0_0_64^3_profiles.pkl', \
      'bondi_multizone_050823_bflux0_0_64^3_nojit_profiles.pkl']]
  listOfLabels = ['n=1', 'n=3_32^3', 'n=3', 'n=3_96^3', 'n=3_128^3', 'HD+jit', 'HD+nojit']
  n_zones_list = [1, 3, 3, 3, 3, 3, 3]
  plot_dir = '../plots/051523_strongfield'
  '''

  # 2c) n=8
  '''
  listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050123_bflux0_2e-8_32^3_n8_profiles.pkl', 'bondi_multizone_050223_bflux0_2e-8_64^3_n8_profiles.pkl', 'bondi_multizone_050423_bflux0_2e-8_96^3_n8_test_faster_rst_profiles2.pkl']]
  listOfLabels = ['jit0.1_32^3', 'jit0.1_64^3', 'jit0.1_96^3']
  n_zones_list = [8, 8, 8]
  plot_dir = '../plots/051523_n8'
  '''

  # 2d) weak field test (n=3)
  '''
  listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050123_onezone_bflux0_1e-8_2d_newrs_profiles.pkl', 'bondi_multizone_050123_bflux0_1e-8_2d_newrs_profiles.pkl', 'bondi_multizone_050423_bflux0_1e-8_2d_newrs_no_short_t_out_profiles.pkl']]
  listOfLabels = ['n=1', 'n=3','n=3_noshort']
  n_zones_list = [1, 3, 3]
  plot_dir = '../plots/050223_n=1vsn=3'
  '''

  # colors, linestyles, directory
  colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']
  linestyles=['-',':',':',':',':', '--', ':']
  os.makedirs(plot_dir, exist_ok=True)

  for quantity in ['beta', 'Mdot', 'rho', 'u', 'T', 'abs_u^r', 'abs_u^phi', 'abs_u^th', 'u^r', 'u^phi', 'u^th']: 
    output = plot_dir+"/profile_"+quantity+".pdf"
    plotProfiles(listOfPickles, quantity, output=output, cycles_to_average=20, color_list=colors, linestyle_list=linestyles, label_list=listOfLabels, rescale=False, \
    n_zones_list=n_zones_list, trim_zone=True, flip_sign=(quantity in ['u^r']))
