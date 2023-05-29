import matplotlib.pyplot as plt
import numpy as np
import bondi_analytic as bondi
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os
import pdb

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
    profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'Omega':
    quantity_index_numerator = dictionary['quantities'].index('u^phi')
    quantity_index_denominator = dictionary['quantities'].index('u^t')
    numerator_list = [profile_list[quantity_index_numerator] for profile_list in dictionary['profiles']]
    denominator_list = [profile_list[quantity_index_denominator] for profile_list in dictionary['profiles']]
    profiles = [[np.array(list[quantity_index_numerator])/np.array(list[quantity_index_denominator]) for list in sublist] for sublist in dictionary['profiles']]
  else:
    quantity_index = dictionary['quantities'].index(quantity)
    profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
  return profiles, invert

def plotProfiles(listOfPickles, quantity, output=None, colormap='turbo', color_list=None, linestyle_list=None, figsize=(8,6), flip_sign=False, show_divisions=True, zone_time_average_fraction=0, 
  xlabel=None, ylabel=None, xlim=None, ylim=None, label_list=None, fig_ax=None, formatting=True, finish=True, rescale=False, rescale_radius=10, rescale_value=1, cycles_to_average=1, trim_zone=True):

  if isinstance(listOfPickles, str):
    listOfPickles = [listOfPickles]

  if label_list is None:
    label_list = [None]*len(listOfPickles)
  if color_list is None:
    color_list = [None]*len(listOfPickles)
  if linestyle_list is None:
    linestyle_list = [None]*len(listOfPickles)
  times_list = [None]*len(listOfPickles)

  #Changes some defaults.
  matplotlib_settings()

  #If you want, provide your own figure and axis.  Good for multipanel plots.
  if fig_ax is None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
  else:
    fig, ax = fig_ax

  #Profiles are pre-computed.
  #See ../compute_scripts/computeProfiles.py for how this file is formatted.

  for sim_index in range(len(listOfPickles)):
    with open(listOfPickles[sim_index], 'rb') as openFile:
      D = pickle.load(openFile)

    radii = D['radii']
    #Formula that produces the zone number of a given run.  It would have been better to have this in some other file though.
    n_zones = D['nzone']
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
      last_time = 0
      for zone in range(n_zones):
        matchingIndices = np.where(zone_number_sequence == zone)[0]
        if zone in [0,n_zones-1]:
          #Tricky edge case: there are half as many instances of the zones at the ends.
          cycles_to_average_halved = int(np.ceil(cycles_to_average/2))
        else: # keep original cycles_to_average
          cycles_to_average_halved = cycles_to_average
        indicesToAverage = matchingIndices[-np.min([len(matchingIndices),cycles_to_average_halved]):]
        print(sim_index, zone, len(matchingIndices))

        #Before moving on, we're going to average profiles within each individual run.
        selectedProfiles = [profiles[i] for i in indicesToAverage]
        selectedProfileTimes = [D['times'][i] for i in indicesToAverage]
        usableProfiles = []
        for run_index in range(len(selectedProfiles)):
          t = selectedProfileTimes[run_index]
          if t[-1] > last_time:
            last_time = t[-1]
          if len(t) > 1: # If there's only 1 output, skip this run
            delt = np.gradient(t)
            averaging_mask = t >= t[-1] - (t[-1]-t[0])*zone_time_average_fraction

            #Always include the last snapshot. Specify this with zone_time_average_fraction == 0.
            if np.sum(averaging_mask) == 0:
              averaging_mask[-1] = True

            #An average, taking care to weight different timesteps appropriately.  There's an annoying squeeze and transpose here.
            integrand = np.transpose(np.squeeze(np.array(selectedProfiles[run_index]))[averaging_mask])
            usableProfiles.append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))
          else: # delete the history of the skipped run
            indicesToAverage = np.delete(indicesToAverage, run_index)
            #selectedProfiles = np.delete(selectedProfiles, run_index)
            #selectedProfileTimes = np.delete(selectedProfileTimes, run_index)

        times_list[sim_index] = last_time

        #Now, given an average within each run, average each separate run.
        plottable = np.mean(usableProfiles, axis=0)

        #Flip the quantity upside-down, usually for inv_beta.
        if invert:
          plottable = 1.0 / plottable

        #Next, optionally mask out regions that likely have wall glitches by only taking the central half of the radii
        finalMatchingIndex = indicesToAverage[-1]
        n_radii = len(radii[finalMatchingIndex])
        mask = np.full(n_radii, True, dtype=bool)
        if n_zones > 1:
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
      ax.plot(r_plot[order], values_plot[order], color=color_list[sim_index], ls=linestyle_list[sim_index], lw=2)
    else:
      #DEPRECATED:  Probably will not run.

      #Plot every iteration of a given run.  Recommended for examining a single run.  Rescaling not supported because you may be asked to interpolate outside the zone.
      for zone in range(n_profiles):
        valuesToColors = plt.cm.get_cmap(colormap)
        color = valuesToColors(zone/n_profiles)
        plottable = profiles[finalMatchingIndex]
        if invert:
          plottable = 1.0 / plottable
        ax.plot(radii[zone], plottable*(-1)**int(flip_sign), color=color, ls=linestyle_list[sim_index], lw=2)

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
    for sim_index in range(len(listOfPickles)):
      ax.plot([], [], color=color_list[sim_index], lw=2, label=label_list[sim_index], ls=linestyle_list[sim_index]) # +', t={:.2g}'.format(times_list[sim_index])
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
  try:
    r_sonic = D["r_sonic"]
  except:
    r_sonic = np.sqrt(1e5)
  analytic_sol = bondi.get_quantity_for_rarr(r_bondi, quantity, rs=r_sonic)
  if analytic_sol is not None:
    ax.plot(r_bondi, analytic_sol, color='slategrey',label='bondi analytic', lw=3, ls='-', zorder=-100)

  #Either show or save.
  if finish:
    if output is None:
      fig.show()
    else:
      fig.savefig(output)
      plt.close(fig)

if __name__ == '__main__':

  #TESTING
  
  '''
  listOfPickles = ['../data_products/bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4_profiles_all.pkl']
  listOfLabels = ['n=1']
  listOfPickles = ['../data_products/bondi_multizone_050423_bflux0_1e-8_2d_n4_profiles_all.pkl']
  listOfLabels = ['n=4']
  '''

  # 2a) weak field test (n=4)
  #listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4_profiles_all.pkl', 'bondi_multizone_050423_bflux0_1e-8_2d_n4_profiles_all.pkl']]
  #listOfLabels = ['n=1', 'n=4']
  #plot_dir = '../plots/052923_weakfield'
  #avg_frac=0
  #cta=1

  # 2b) strong field test
  '''
  listOfPickles = ['../data_products/'+dirname for dirname in \
      ['bondi_multizone_050123_onezone_bflux0_1e-4_64^3_profiles_all.pkl', \
      'bondi_multizone_042723_bflux0_1e-4_32^3_profiles_all.pkl', \
      'bondi_multizone_042723_bflux0_1e-4_64^3_profiles_all.pkl', \
      'bondi_multizone_050123_bflux0_1e-4_96^3_profiles_all.pkl', \
      'bondi_multizone_050523_bflux0_1e-4_128^3_n3_noshort_profiles_all.pkl', \
      'bondi_multizone_050123_bflux0_0_64^3_profiles_all.pkl', \
      'bondi_multizone_050823_bflux0_0_64^3_nojit_profiles_all.pkl']]
  listOfLabels = ['n=1', 'n=3_32^3', 'n=3', 'n=3_96^3', 'n=3_128^3', 'HD+jit', 'HD+nojit']
  n_zones_list = [1, 3, 3, 3, 3, 3, 3]
  plot_dir = '../plots/052923_strongfield'
  avg_frac=0.3
  cta=75
  '''

  # 2c) n=8
  listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050123_bflux0_2e-8_32^3_n8_profiles_all.pkl', 'bondi_multizone_050223_bflux0_2e-8_64^3_n8_profiles_all.pkl', 'bondi_multizone_050423_bflux0_2e-8_96^3_n8_test_faster_rst_profiles_all.pkl', 'production_runs/bondi_bz2e-8_1e8_profiles_all.pkl', '/production_runs/bondi_bz2e-8_1e8_96_profiles_all.pkl']]
  listOfLabels = ['32^3', '64^3', '96^3', '64^3_C', '96^3_C']
  plot_dir = '../plots/052923_n8'
  avg_frac=0.5
  cta=40

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
    #output = None
    plotProfiles(listOfPickles, quantity, output=output, zone_time_average_fraction=avg_frac, cycles_to_average=cta, color_list=colors, linestyle_list=linestyles, label_list=listOfLabels, rescale=False, \
    trim_zone=True, flip_sign=(quantity in ['u^r']))
