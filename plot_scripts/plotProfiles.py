import matplotlib.pyplot as plt
import numpy as np
import bondi_analytic as bondi
from matplotlib_settings import *
from ylabel_dictionary import *
import pickle
import os
import pdb
from scipy.ndimage import uniform_filter1d

import glob
import h5py

def readQuantity(dictionary, quantity):

  invert = False
  if quantity == 'beta':
    #It's better to use inverse beta, if we have it.
    # Hyerin (08/09/23) modified such that Pg and Pb are averaged separately.
    try:
      quantity_index = dictionary['quantities'].index('inv_beta')
      invert = True
    except:
      print("inv_beta doesn't exist, so we will stick with beta.")
      quantity_index = dictionary['quantities'].index(quantity)
    profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'eta':  #TODO this needs to be corrected such that Mdot and Edot averaged over separately first
    quantity_index_numerator = dictionary['quantities'].index('Edot')
    quantity_index_denominator = dictionary['quantities'].index('Mdot')
    profiles = [[1.-np.array(list[quantity_index_numerator])/np.array(list[quantity_index_denominator]) for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'etaMdot':
    quantity_index_numerator = dictionary['quantities'].index('Edot')
    quantity_index_denominator = dictionary['quantities'].index('Mdot')
    profiles = [[np.array(list[quantity_index_denominator])-np.array(list[quantity_index_numerator]) for list in sublist] for sublist in dictionary['profiles']]
  #elif quantity == 'Omega':
    #quantity_index_numerator = dictionary['quantities'].index('Omega')
    #quantity_index_denominator = dictionary['quantities'].index('u^t')
    #profiles = [[abs(np.array(list[quantity_index_numerator])/np.array(list[quantity_index_denominator])) for list in sublist] for sublist in dictionary['profiles']]
  elif quantity == 'Pg':
    try:
      quantity_index = dictionary['quantities'].index('Pg')
      profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
    except:
      gam = 5./3. # TODO: I'm currently assuming that the adiabatic index is always this.
      quantity_index = dictionary['quantities'].index('u')
      profiles = [[np.array(list[quantity_index])*(gam-1.) for list in sublist] for sublist in dictionary['profiles']]
  else:
    quantity_index = dictionary['quantities'].index(quantity)
    profiles = [[list[quantity_index] for list in sublist] for sublist in dictionary['profiles']]
  return profiles, invert

def assignTimeBins(D, profiles, ONEZONE=False, num_time_chunk=4, zone_time_average_fraction=0.5, factor=2):
  # Hyerin (06/20/23) assign to different time bins
  n_profiles = len(profiles)

  t_first = np.min(D["times"][0])
  t_last = np.max(D["times"][-1])
  tDivList = np.array([t_first+(t_last-t_first)/np.power(factor,i+1) for i in range(num_time_chunk)])
  tDivList = tDivList[::-1] # in increasing time order
  n_zones = D['nzone']
  try: n_zones_eff = D['nzone_eff']
  except : n_zones_eff = n_zones

  #if 'moving_rin' in D['runName'] or 'combine_outer' in D['runName']: # temporary Hyerin (07/31/23)
  #  n_zones_eff -= 2
  radii = D['radii']
  usableProfiles = [[[] for _ in range(num_time_chunk)] for _ in range(n_zones_eff)] # (n_zones, num_time_chunk) dimension
  r_save = [None]*n_zones_eff
  num_save = np.zeros((n_zones_eff,num_time_chunk))
  try: base = D["base"]
  except: base = 8

  for i,profile in enumerate(profiles):
    times = D["times"][i]
    run_idx = D["runIndices"][i]
    if n_zones_eff == 1: iteration = 0
    else: iteration = np.maximum(np.ceil(run_idx/(n_zones_eff-1)),1)
    if 1: #iteration % 2 == 1 or run_idx % (n_zones-1)==0: # selecting only outwards direction
        if len(times)<1:
            #pdb.set_trace()
            continue # skip this iteration
        sorting = np.argwhere(tDivList<times[0]) # put one annulus run in the same time bin, even if one zone run corresponds to multiple bins
        bin_num = sorting[-1,0] if len(sorting)>0 else None
        zone_num = n_zones_eff-1 -int(np.floor(np.log(radii[i][0])/np.log(base))-(base<2))
       
        # taken from above
        if len(times) > 1 and bin_num is not None or ONEZONE: # If there's only 1 output, skip this run
          delt = np.ones(len(times)) #np.gradient(times)
          if ONEZONE:
            for bin_num in range(num_time_chunk):
              if bin_num == num_time_chunk-1:
                averaging_mask = times > tDivList[bin_num]
              else:
                averaging_mask = (times > tDivList[bin_num]) & (times < tDivList[bin_num+1])
              integrand = np.transpose(np.squeeze(np.array(profile))[averaging_mask])
              usableProfiles[zone_num][bin_num].append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))
          else:
            averaging_mask = times >= times[-1] - (times[-1]-times[0])*zone_time_average_fraction

            #Always include the last snapshot. Specify this with zone_time_average_fraction == 0.
            if np.sum(averaging_mask) == 0:
              averaging_mask[-1] = True

            #An average, taking care to weight different timesteps appropriately.  There's an annoying squeeze and transpose here.
            #if quantity == "Mdot": # only selecting positive mdots
                #profile = np.array(profile)
                #profile[profile<0] = 0
            integrand = np.transpose(np.squeeze(np.array(profile))[averaging_mask])
            usableProfiles[zone_num][bin_num].append(np.sum(integrand*delt[averaging_mask], axis=1) / np.sum(delt[averaging_mask]))

            num_save[zone_num][bin_num] += 1 # record how many saves per bin

          if r_save[zone_num] is None:
            r_save[zone_num] = radii[i]

  tDivList = np.append(tDivList,t_last)
  return tDivList, usableProfiles, r_save, num_save

def plotProfiles(listOfPickles, quantity, output=None, colormap='turbo', color_list=None, linestyle_list=None, figsize=(8,6), flip_sign=False, show_divisions=True, show_rb=False, zone_time_average_fraction=0, 
  xlabel=None, ylabel=None, xlim=None, ylim=None, label_list=None, fig_ax=None, formatting=True, finish=True, rescale=False, rescale_radius=10, rescale_value=1, cycles_to_average=1, trim_zone=True, show_init=False, show_gizmo=False, show_bondi=False, show_rscale=False, show_mdotinout=False, num_time_chunk=4, boxcar_factor=0, average_factor=2):

  if isinstance(listOfPickles, str):
    listOfPickles = [listOfPickles]

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
    try: n_zones_eff = D['nzone_eff']
    except: n_zones_eff = n_zones
    #if 'moving_rin' in D['runName'] or 'combine_outer' in D['runName']: # temporary Hyerin (07/31/23)
    #  n_zones_eff -= 2
    if n_zones_eff > 1:
      #zone_number_sequence = np.array([np.abs(np.abs(n_zones-1 - (i % (2*n_zones-2)))-(n_zones-1)) for i in range(len(radii))])
      try:
          base = D["base"]
      except:
          base = 8
      try:
        zone_number_sequence = np.array(D["zones"])
      except:
        zone_number_sequence = np.array([n_zones_eff-1 - int(np.floor(np.log(radii[i][0])/np.log(base))) for i in range(len(radii))])
    else:
      zone_number_sequence = np.full(len(radii), 0)

    if cycles_to_average > 0:
      if color_list is None:
        color_list = [None]*len(listOfPickles)
      if label_list is None:
        label_list = [None]*len(listOfPickles)
      profiles, invert = readQuantity(D, quantity)
      profiles = profiles
      n_profiles = len(profiles)

      r_plot = np.array([])
      values_plot = np.array([])
      init_r = np.array([])
      init_plot = np.array([])
      #Only plot the very last iteration.  Recommended for comparing different initial conditions.
      rescalingFactor = 1.0
      if rescale:
        #Find rescaling factor.
        #for r_set, profile_set in zip(radii,profiles):
        #  if (rescale_radius >= r_set[0]) & (rescale_radius <= r_set[-1]):
        #    rescalingFactor = rescale_value / np.interp(np.log10(rescale_radius), np.log10(r_set), profile_set)
        rescalingFactor = 1./rescale_value
      last_time = 0
      for zone in range(n_zones_eff):
        matchingIndices = np.where(zone_number_sequence == zone)[0]
        if zone in [0,n_zones_eff-1]:
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
        if show_init and (quantity == "rho" or quantity == "T" or quantity == "beta"):
            # show initial conditions
            mask = np.full(len(radii[matchingIndices[0]]), True, dtype=bool)
            if zone > 0:
              mask[-int(n_radii/2):] = False
            init_r = np.concatenate([init_r,radii[matchingIndices[0]][mask]])
            init_plot = np.concatenate([init_plot,profiles[matchingIndices[0]][0][mask]])
            

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
        if n_zones_eff > 1:
          if trim_zone:
            if zone > 0:
              mask[-int(n_radii/4):] = False
            if zone < n_zones_eff - 1:
              mask[:int(n_radii/4)] = False

        r_plot = np.concatenate([r_plot,radii[finalMatchingIndex][mask]])
        values_plot = np.concatenate([values_plot,rescalingFactor*plottable[mask]*(-1)**int(flip_sign)])

      r_plot = np.squeeze(np.array(r_plot))
      values_plot = np.squeeze(np.array(values_plot))
      order = np.argsort(r_plot)
      if label_list is None: label = '__nolegend__'
      else: label = label_list[sim_index]
      ax.plot(r_plot[order], values_plot[order], color=color_list[sim_index], ls=linestyle_list[sim_index], lw=2,label=label)
      if show_init and (quantity == "rho" or quantity == "T" or quantity=="beta" or quantity=='u^r'):
        order = np.argsort(init_r)
        if invert: ax.plot(np.array(init_r[order]), 1./init_plot[order], color=color_list[sim_index], ls=':', lw=1, alpha=1)
        else: ax.plot(np.array(init_r[order]), init_plot[order], color=color_list[sim_index], ls=':', lw=1, alpha=1)

    else:
      # HYERIN: split into time chunks
      if 'onezone' in listOfPickles[sim_index]: ONEZONE = True
      else: ONEZONE = False
      
      #average_factor = 2 #np.sqrt(2) #2 #  1.1 # 
      rescalingFactor = 1.0
      if rescale:
        #Find rescaling factor.
        rescalingFactor = 1./rescale_value
      if quantity == 'eta':
          profiles, invert = readQuantity(D, 'Edot')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
          profiles, _ = readQuantity(D, 'Mdot')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
          #usableProfiles = 1.-usableProfiles_num/usableProfiles_den
      elif quantity == 'etaMdot':
          profiles, invert = readQuantity(D, 'Edot')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
          profiles, _ = readQuantity(D, 'Mdot')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
      elif quantity == 'u^r':
          profiles, invert = readQuantity(D, 'Mdot')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
          profiles, _ = readQuantity(D, 'rho')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
      #elif quantity == 'beta':
      #    profiles, invert = readQuantity(D, 'Pg')
      #    tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
      #    profiles, _ = readQuantity(D, 'Pb')
      #    _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
      elif quantity == 'phib':
          profiles, invert = readQuantity(D, 'Phib')
          tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
          profiles, _ = readQuantity(D, 'Mdot')
          _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
      else:
          profiles, invert = readQuantity(D, quantity)
          tDivList, usableProfiles, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor)
      
      if color_list is None: colors=plt.cm.gnuplot(np.linspace(0.9,0.3,num_time_chunk))
      else: colors= [color_list[sim_index]]
      for b in range(num_time_chunk):
        r_plot = np.array([])
        values_plot = np.array([])
        print("{}: t={:.3g}-{:.3g}, # of run summed is {}".format(b,tDivList[b],tDivList[b+1], num_save[:,b]))
        for zone in range(n_zones_eff):
          #Now, given an average within each run, average each separate run.
          if quantity == 'eta':
            plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
            if zone == n_zones_eff-1: # r=10
                i10 = np.argmin(abs(r_save[zone]-10))
                Mdot_save = plottable_den[i10]
            plottable = plottable_den- plottable_num #+ 4.5e-3
            #plottable = np.mean(usableProfiles[zone][b],axis=0)
            invert = False
          elif quantity == 'etaMdot':
            plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
            plottable = plottable_den - plottable_num
            invert = False
          elif quantity == "u^r":
            plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)*(4.*np.pi*r_save[zone]**2) # density averaged
            plottable = -plottable_num/plottable_den
            invert = False
          elif "Omega" in quantity:
            plottable = (np.mean(usableProfiles[zone][b], axis=0) * np.power(r_save[zone],3./2)) # normalize by Omega_K
          elif "phib" in quantity:
            plottable = np.mean(usableProfiles_num[zone][b], axis=0)
            plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
            if zone == n_zones_eff-1: # r=10
                i10 = np.argmin(abs(r_save[zone]-10))
                Mdot_save = plottable_den[i10]
          #elif quantity == "beta":
          #  plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
          #  plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
          #  plottable = plottable_num/plottable_den 
          #  invert = False
          else:
            plottable = np.mean(usableProfiles[zone][b], axis=0)

          #Flip the quantity upside-down, usually for inv_beta.
          if invert:
            plottable = 1.0 / plottable

          #Next, optionally mask out regions that likely have wall glitches by only taking the central half of the radii
          if 'moving_rin' in D['runName']: # temporary Hyerin (07/31/23)
            n_radii = int(len(r_save[n_zones_eff-1])*2/(n_zones+1))
            
            mask = np.full(len(r_save[zone]), True, dtype=bool)
            if trim_zone:
                if zone != n_zones_eff-1:
                    mask[:]=False
          else:
            n_radii = len(r_save[n_zones_eff-1])#len(r_save[zone]) # only take smallest annuli'ls n_radii

            mask = np.full(len(r_save[zone]), True, dtype=bool)
            if n_zones_eff > 1:
              if trim_zone:
                if zone > 0:
                  mask[-int(n_radii/4):] = False
                if zone < n_zones_eff - 1:
                  mask[:int(n_radii/4)] = False

          r_plot = np.concatenate([r_plot,r_save[zone][mask]])
          values_plot = np.concatenate([values_plot,rescalingFactor*plottable[mask]*(-1)**int(flip_sign)])

        r_plot = np.squeeze(np.array(r_plot))
        values_plot = np.squeeze(np.array(values_plot))
        if quantity == "eta":
            values_plot /= Mdot_save # divide by  Mdot at r=10
        if quantity == "phib":
            values_plot /= np.sqrt(Mdot_save)
        order = np.argsort(r_plot)
        if label_list is None: label = 't={:.5g} - {:.5g}'.format(tDivList[b],tDivList[b+1])
        else: label = label_list[sim_index]
        if boxcar_factor == 0:
            boxcar_avged = values_plot[order]
        else:
            boxcar_avged = uniform_filter1d(values_plot[order], size=n_radii//boxcar_factor) # (07/12/23) boxcar averaging
        ax.plot(r_plot[order], boxcar_avged, color=colors[b], ls=linestyle_list[sim_index], lw=2, label=label) 
        if (num_time_chunk == 1): ax.plot(r_plot[order], -boxcar_avged, color=colors[b], ls=':', lw=2) 

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
    if show_divisions:
      divisions = []
      #for zone in range(n_zones):
      #  divisions.append(radii[zone][-1])
      #  if (zone == n_zones-1) | (zone == n_zones-2):
      #    divisions.append(radii[zone][0])
      divisions = [base**i for i in range(n_zones+2)]
      for div in divisions:
        ax.plot([div]*2, ax.get_ylim(), alpha=0.2, color='grey', lw=1)

  try:
    r_sonic = D["r_sonic"]
  except:
    r_sonic = np.sqrt(1e5)

  if show_rb:
    ax.axvline(r_sonic**2, color='grey', lw=1, alpha=0.2)
  if show_bondi:
    # Bondi analytic overplotting
    xlim = ax.get_xlim()
    r_bondi = np.logspace(np.log10(max(2,xlim[0])), np.log10(xlim[1]), 50)
    analytic_sol = bondi.get_quantity_for_rarr(r_bondi, quantity, rs=r_sonic)
    if analytic_sol is not None and not rescale:
      if quantity == "rho": label = "Bondi analytic"
      else: label = "__nolegend__"
      ax.plot(r_bondi, analytic_sol, color='slategrey',label=label, lw=6, ls='-', zorder=-100,alpha=0.5)
    rb = 1e5
    #rho0 = bondi.get_quantity_for_rarr([1e8], quantity, rs=r_sonic)
    #if 'rho' in quantity:
    #    ax.plot(r_bondi,rho0 * (r_bondi + rb) / r_bondi,'k:')
  if show_gizmo:
    dat_gz=np.loadtxt("/n/holylfs05/LABS/bhi/Users/hyerincho/grmhd//data/gizmo/031623_100Myr/dat.txt")
    r_gizmo=dat_gz[:,0]
    rho_gizmo = dat_gz[:,1]
    T_gizmo=dat_gz[:,2]
    vr_gizmo=dat_gz[:,3]
    to_plot=None
    min_rgizmo = r_gizmo[0]
    if quantity=='rho':
      to_plot = rho_gizmo
    elif quantity=='u':
      to_plot = T_gizmo*rho_gizmo*3/2
    elif quantity=='Pg':
      to_plot = T_gizmo*rho_gizmo
    elif quantity=='T':
      to_plot = T_gizmo
    else:
      r_gizmo=[] ;to_plot = None #[]
    if to_plot is not None:
        ax.plot(r_gizmo,to_plot,'b-',lw=5,label='GIZMO', zorder=-100,alpha=0.3)

    # extend inwards with expected Bondi solution
    if len(r_gizmo)>1:
        r_bondi = np.logspace(np.log10(2), np.log10(min_rgizmo), 50)
    else:
        xlim = ax.get_xlim()
        r_bondi = np.logspace(np.log10(max(2,xlim[0])), np.log10(xlim[1]), 50)
    label = "__nolegend__"
    if quantity != "Mdot":
        analytic_sol = bondi.get_quantity_for_rarr(r_bondi, quantity, rs=r_sonic)
        if quantity == "rho" or quantity == "T":
            analytic_sol *= to_plot[0]/analytic_sol[-1]
            if quantity == "rho": label = "Bondi extension"
    else:
        rho = bondi.get_quantity_for_rarr([min_rgizmo],'rho',rs=r_sonic)
        Mdot = bondi.get_quantity_for_rarr([min_rgizmo], 'Mdot', rs=r_sonic)
        analytic_sol = [Mdot/rho[0] * rho_gizmo[0]]*len(r_bondi)
    if analytic_sol is not None: ax.plot(r_bondi,analytic_sol,'b-.',label=label,lw=5, zorder=-100,alpha=0.2)

  elif show_rscale!=False:
    # show density scalings
    if "rho" in quantity and (show_rscale==True or "rho" in show_rscale): # and "rot" in dirtag:
        rarr= np.logspace(np.log10(2),np.log10(r_sonic),20) #*1000
        factor=1e-8*1e5/r_sonic**2#7e-7
        ax.plot(rarr,np.power(rarr/1e3,-1)*factor,'g-',alpha=0.5,lw=3,label=r'$r^{-1}$')
        #ax.plot(rarr,np.power(rarr/1e3,-1.5)*factor,'g:',alpha=0.3,lw=10,label=r'$r^{-1.5}$')
        #rb=1e5
        #rho0=np.power(3e-6,1.5)
        #ax.plot(rarr,rho0 * (rarr + rb) / rarr,'k-')
        #ax.plot(rarr,np.power(rarr/1e3,-1/2)*factor,'g:',alpha=0.3,lw=10,label=r'$r^{-1/2}$')
    if quantity == "beta" and (show_rscale==True or "beta" in show_rscale): # and "rot" in dirtag:
        rarr= np.logspace(np.log10(2),np.log10(100),20) #*1000
        factor=1e7/r_sonic**2#
        ax.plot(rarr,np.power(rarr/1e3,3/2.)*factor,'g-',alpha=0.5,lw=3,label=r'$r^{3/2}$')
    if quantity == "phib" and (show_rscale==True or "phib" in show_rscale): # and "rot" in dirtag:
        rarr= np.logspace(np.log10(1e2),np.log10(1e5),20) #*1000
        factor=1/2#
        ax.plot(rarr,np.power(rarr,1)*factor,'g-',alpha=0.5,lw=3,label=r'$r^{1}$')
  # legends
  if formatting:
    for sim_index in range(len(listOfPickles)):
      if cycles_to_average > 0: ax.plot([], [], color=color_list[sim_index], lw=2, label=label_list[sim_index], ls=linestyle_list[sim_index]) # +', t={:.2g}'.format(times_list[sim_index])
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

  #TESTING
  
  '''
  listOfPickles = ['../data_products/bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4_profiles_all.pkl']
  listOfLabels = ['n=1']
  listOfPickles = ['../data_products/bondi_multizone_050423_bflux0_1e-8_2d_n4_profiles_all.pkl']
  listOfLabels = ['n=4']
  '''
  
  # default plotting setting. change below if needed
  xlim=None
  show_rscale=False
  show_div=True
  show_rb=False
  boxcar_factor=0
  
  # 1a) Bondi
  '''
  listOfPickles = ['../data_products/bondi_multizone_030723_bondi_128^3_profiles_all.pkl']
  listOfLabels = [''] #'GIZMO, no extg', 
  plot_dir = '../plots/052923_bondi'
  avg_frac=0.
  cta=1
  xlim=[2,1e8]
  '''

  # 1b&c) GIZMO
  '''
  listOfPickles = ['../data_products/production_runs/gizmo_extg_1e8_profiles_all.pkl', '../data_products/bondi_multizone_052523_gizmo_n8_64^3_noshock_profiles_all.pkl']
  listOfLabels = ['Ext. Grav.', 'No Ext. Grav.']
  plot_dir = '../plots/052923_gizmo'
  avg_frac=0.5
  cta=10
  num_time_chunk=-1
  show_div=False
  show_rb=True

  xlim=[2,4e9]
  colors = ['tab:blue','tab:green', 'tab:red', 'k','r', 'b',  'g', 'm', 'c', 'y']
  '''

  # 2a) weak field test (n=4)
  '''
  listOfPickles = ['../data_products/'+dirname for dirname in ['bondi_multizone_050423_onezone_bflux0_1e-8_2d_n4_profiles_all2.pkl', 'bondi_multizone_050423_bflux0_1e-8_2d_n4_profiles_all.pkl']]
  listOfLabels = ['n=1', 'n=4']
  plot_dir = '../plots/052923_weakfield'
  avg_frac=0
  cta=1
  xlim=[2,3.5e4]
  colors = ['k','r', 'b',  'g', 'm', 'c', 'y']
  '''

  # 2b) strong field test
  '''
  dictOfAll = {}
  dictOfAll['oz'] = ['061623_ozrst_onezone_profiles_all2.pkl', 'k']
  #dictOfAll['32_r^1'] = ['062623_n3_1_profiles_all2.pkl', 'y']
  dictOfAll['beta1e0'] = ['071023_n3_beta01_profiles_all2.pkl', 'lightgreen']
  listOfPickles = []
  listOfLabels =[]
  colors = []
  for key,value in dictOfAll.items():
      listOfPickles += ['../data_products/'+value[0]]
      listOfLabels += [key]
      colors += [value[1]]
  plot_dir = '../plots/062023_strongfield'
  avg_frac=0.1 #0 #
  cta=0 #2 #
  show_rscale=False
  num_time_chunk=1
  xlim=[2,4.5e3]
  '''

  # 2c) n=8
  '''
  dictOfAll = {}
  #dictOfAll['64_rst_b64'] = ['080823_rst64_testb64_profiles_all2.pkl','crimson']
  dictOfAll['64_rst_longtin'] = ['081723_rst64_longtin_profiles_all2.pkl', 'c']
  dictOfAll['128'] = ['production_runs/072823_beta01_128_profiles_all2.pkl', 'm']
  #dictOfAll['32_hd'] = ['062623_hd_ur0_profiles_all2.pkl','c']
  listOfPickles = []
  listOfLabels =[]
  colors = []
  for key,value in dictOfAll.items():
      listOfPickles += ['../data_products/'+value[0]]
      listOfLabels += [key]
      colors += [value[1]]
  plot_dir = '../plots/082123_n8'
  avg_frac=0.5
  cta=0 # 40
  num_time_chunk = 1
  xlim=[2,2e8]
  show_rscale=False #True
  boxcar_factor=4 #2
  '''

  # test
  #listOfPickles = ['../data_products/production_runs/072823_beta01_128_profiles_all2.pkl']
  #listOfPickles = ['../data_products/production_runs/072823_beta01_onezone_profiles_all2.pkl']
  #listOfPickles = ['../data_products/082223_rst64_longtin4_profiles_all2.pkl']
  listOfPickles = ['../data_products/082423_n4_profiles_all2.pkl']
  #listOfPickles = ['../data_products/091123_n4_128_profiles_all2.pkl']

  listOfLabels = None
  plot_dir = "../plots/test"
  cta=0 # time evolution
  avg_frac=0.5
  show_rscale=True
  num_time_chunk=4 #8 #4 #2 #
  boxcar_factor=0 #4 #2 #
  colors=None
  xlim=[2,1e4] #2e8] #
  show_div=False # temp

  # colors, linestyles, directory
  linestyles=['-','-',':',':',':', '--', ':']
  os.makedirs(plot_dir, exist_ok=True)

  for quantity in ['eta', 'etaMdot', 'beta', 'Edot', 'Mdot', 'rho', 'u', 'T', 'abs_u^r', 'abs_u^phi', 'abs_u^th', 'u^r', 'u^phi',  'u^th']: #'abs_Omega', 'K', 'Pg', 
    output = plot_dir+"/profile_"+quantity+".pdf"
    #output = None
    ylim = [None,[1e-4,10]][(quantity in ['Mdot'])] # [1e-3,10] if Mdot, None otherwise
    ylim = [ylim,[1e-4,2]][(quantity in ['abs_Omega'])] #
    plotProfiles(listOfPickles, quantity, output=output, zone_time_average_fraction=avg_frac, cycles_to_average=cta, color_list=colors, linestyle_list=linestyles, label_list=listOfLabels, rescale=False, \
    trim_zone=True, flip_sign=(quantity in ['u^r']), xlim=xlim, ylim=ylim ,show_gizmo=("gizmo" in plot_dir), show_rscale=show_rscale, num_time_chunk=num_time_chunk, boxcar_factor=boxcar_factor, show_divisions=show_div, show_rb=show_rb, average_factor=2)
