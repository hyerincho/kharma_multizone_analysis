import matplotlib.pyplot as plt
import numpy as np
import bondi_analytic as bondi
from matplotlib_settings import *
import pickle
import os
import pdb

#TODO:  There should be a file in each folder that tells you how many zones there are, and other useful information.

def plotProfiles(listOfPickles, quantity, output=None, colormap='turbo', color_list=None, linestyle_list=None, figsize=(8,6), n_zones_list=[7], flip_sign=False,
  xlabel=None, ylabel=None, xlim=None, ylim=None, label_list=None, fig_ax=None, formatting=True, finish=True, rescale=False, rescaleRadius=10, rescaleValue=1, cyclesToAverage=1, trimZone=True):

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

    invert = False
    if quantity == 'beta':
      #It's better to use inverse beta, if we have it.
      try:
        quantity_index = D['quantities'].index('inv_beta')
        invert = True
      except:
        print("inv_beta doesn't exist, so we will stick with beta.")
        quantity_index = D['quantities'].index(quantity)
    else:
      quantity_index = D['quantities'].index(quantity)
    profiles = [profile_list[quantity_index] for profile_list in D['profiles']]
    n_profiles = len(profiles)
    if cyclesToAverage > 0:
      r_plot = np.array([])
      values_plot = np.array([])
      #Only plot the very last iteration.  Recommended for comparing different initial conditions.
      rescalingFactor = 1.0
      if rescale:
        #Find rescaling factor.
        for r_set, profile_set in zip(radii,profiles):
          if (rescaleRadius >= r_set[0]) & (rescaleRadius <= r_set[-1]):
            rescalingFactor = rescaleValue / np.interp(np.log10(rescaleRadius), np.log10(r_set), profile_set)
      for zone in range(n_zones):
        matchingIndices = np.where(zone_number_sequence == zone)[0]
        indicesToAverage = matchingIndices[-np.min([len(matchingIndices),cyclesToAverage]):]
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
        if trimZone:
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

  #TODO:  I recommend doing analytic overplotting here!

  #Formatting
  if formatting:
    if xlabel is None:
      xlabel = 'Radius [$r_g$]'
      ax.set_xlabel(xlabel)
    if ylabel is None:
      ylabel = quantity
      ax.set_ylabel(ylabel)  #TODO:  Make a dictionary that matches quantity names with nice latex.
      ax.set_xscale('log')
      ax.set_yscale('log')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)
    for run_index in range(len(listOfPickles)):
      ax.plot([], [], color=color_list[run_index], lw=2, label=label_list[run_index])
      ax.legend(loc='best', frameon=False)
      fig.tight_layout()

  #Either show or save.
  if finish:
    if output is None:
      fig.show()
    else:
      fig.savefig(output)
      plt.close(fig)

if __name__ == '__main__':

  # 2a) weak field test (n=4)
  '''
  listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4_profiles.pkl', 'bondi_multizone_050423_bflux0_1e-8_2d_n4_profiles.pkl']]
  listOfLabels = ['n=1', 'n=3']
  n_zones_list = [1, 3]
  plot_dir = '../plots/050823_weakfield'
  '''

  # 2b) strong field test
  listOfPickles = ['../data_products/'+dirname for dirname in \
      ['bondi_multizone_050123_onezone_bflux0_1e-4_64^3_profiles.pkl', \
      'bondi_multizone_042723_bflux0_1e-4_32^3_profiles.pkl', \
      'bondi_multizone_042723_bflux0_1e-4_64^3_profiles.pkl', \
      'bondi_multizone_050123_bflux0_1e-4_96^3_profiles.pkl', \
      'bondi_multizone_050523_bflux0_1e-4_128^3_n3_noshort_profiles.pkl', \
      'bondi_multizone_050123_bflux0_0_64^3_profiles.pkl', \
      'bondi_multizone_050823_bflux0_0_64^3_nojit_profiles.pkl']]
  listOfLabels = ['n=1', 'n=3_32^3', 'n=3', 'n=3_96^3', 'n=3_128^3', 'HD+jit', 'HD+nojit']
  n_zones_list = [1, 3, 3, 3, 3, 3, 3]
  plot_dir = '../plots/050223_strongfield'

  # 2c) n=8
  #listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050123_bflux0_2e-8_32^3_n8_profiles.pkl', 'bondi_multizone_050123_bflux0_2e-8_32^3_n8_rot_profiles.pkl', 'bondi_multizone_050223_bflux0_2e-8_64^3_n8_profiles.pkl', 'bondi_multizone_050423_bflux0_2e-8_96^3_n8_test_faster_rst_profiles.pkl']] #, 'bondi_multizone_041823_bondi_n8b8_profiles.pkl']]
  #listOfLabels = ['jit0.1_32^3', 'jit0.01+rot_32^3', 'jit0.1_64^3', 'jit0.1_96^3', 'HD']
  #plot_dir = '../plots/050223_n8'

  # 2d) weak field test (n=3)
  #listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050123_onezone_bflux0_1e-8_2d_newrs_profiles.pkl', 'bondi_multizone_050123_bflux0_1e-8_2d_newrs_profiles.pkl', 'bondi_multizone_050423_bflux0_1e-8_2d_newrs_no_short_t_out_profiles.pkl']]
  #listOfLabels = ['n=1', 'n=3','n=3_noshort']
  #n_zones_list = [1, 3, 3]
  #plot_dir = '../plots/050223_n=1vsn=3'

  # colors, linestyles, directory
  colors = ['k', 'b', 'r','g', 'm', 'c', 'y']
  linestyles=['-',':',':',':',':', '--', ':']
  os.makedirs(plot_dir, exist_ok=True)

  for quantity in ['abs_u^r', 'Mdot', 'rho', 'u', 'T', 'u^r', 'abs_u^phi', 'abs_u^th']: #"beta"]:#, 'b', 
    #TESTING
    #output = plot_dir+"/profile_"+quantity+".pdf"
    output = None
    plotProfiles(listOfPickles, quantity, output=output, cyclesToAverage=1, color_list=colors, linestyle_list=linestyles, label_list=listOfLabels, rescale=False, n_zones_list=n_zones_list, trimZone=True)
