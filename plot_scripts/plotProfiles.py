import matplotlib.pyplot as plt
import numpy as np
import bondi_analytic as bondi
from matplotlib_settings import *
import pickle
import os

#TODO:  There should be a file in each folder that tells you how many zones there are, and other useful information.

def plotProfiles(listOfPickles, quantity, output=None, final_only=False, colormap='turbo', color_list=['k'], figsize=(8,6), n_zones=7, flip_sign=False, 
    show_negatives=True, xlabel=None, ylabel=None, xlim=None, ylim=None):

  if isinstance(listOfPickles, str):
    listOfPickles = [listOfPickles]

  #Changes some defaults.
  matplotlib_settings()

  fig, ax = plt.subplots(1, 1, figsize=figsize)

  #Profiles are pre-computed.
  #See ../compute_scripts/computeProfiles.py for how this file is formatted.
  for run_index in range(len(listOfPickles)):
    with open(listOfPickles[run_index], 'rb') as openFile:
      D = pickle.load(openFile)

    radii = D['radii']
    quantity_index = D['quantities'].index(quantity)
    profiles = [profile_list[quantity_index] for profile_list in D['profiles']]
    n_profiles = len(profiles)
    if final_only:
      #Only plot the very last iteration.  Recommended for comparing different initial conditions.
      for zone in range(n_zones):
        ax.plot(radii[-1-zone], profiles[-1-zone]*(-1)**int(flip_sign), color=color_list[run_index], lw=2)
        if show_negatives:
          ax.plot(radii[-1-zone], -profiles[-1-zone]*(-1)**int(flip_sign), color=color_list[run_index], lw=2, ls=':')
    else:
      #Plot every iteration of a given run.  Recommended for examining a single run.
      for zone in range(n_profiles):
        valuesToColors = plt.cm.get_cmap(colormap)
        color = valuesToColors(zone/n_profiles)
        ax.plot(radii[zone], profiles[zone]*(-1)**int(flip_sign), color=color, lw=2)
        if show_negatives:
          ax.plot(radii[zone], -profiles[zone]*(-1)**int(flip_sign), color=color, lw=2, ls=':')

  #TODO:  I recommend doing analytic overplotting here!

  #Formatting
  if xlabel is None:
    xlabel = 'Radius [$r_g$]'
  ax.set_xlabel(xlabel)
  if ylabel is None:
    ylabel = quantity
  ax.set_ylabel(ylabel)
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  fig.tight_layout()

  #Either show or save.
  if output is None:
    fig.show()
  else:
    fig.savefig(output)
    plt.close(fig)

if __name__ == '__main__':
  dirname="bondi_multizone_010223_bondi_b_n2b8_0"
  listOfPickles = ['../data_products/'+dirname+'_profiles.pkl']
  plot_dir="../plots/"+dirname
  os.makedirs(plot_dir, exist_ok=True)

  for quantity in ['Mdot', 'rho', 'u', 'T', 'u^r', 'u^phi']:
    output = plot_dir+"/profile_"+quantity+".png"
    plotProfiles(listOfPickles, quantity, output=output, final_only=True)
