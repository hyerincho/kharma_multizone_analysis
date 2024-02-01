import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

from plotProfiles import assignTimeBins, readQuantity
from matplotlib_settings import *
from ylabel_dictionary import *

def r_EH(a):
    # event horizon
    a = np.array(a)
    return (1. + np.sqrt(1. - a ** 2))

def eta_BZ6(phi, a, kappa=0.05):
    # Narayan et al (2022) eq (10)
    phi = np.array(phi); a = np.array(a)
    r_h = r_EH(a)
    Omega = a / (2 * r_h)
    return kappa / (4. * np.pi) * phi**2 * Omega**2 * (1. + 1.38 * Omega**2 - 9.2 * Omega**4)

def phi_fit(a):
    # Narayan et al (2022) eq (9)
    return -20.2 * a**3 - 14.9 * a**2 + 34 * a + 52.6

def calc_spin_quantities(pkl, quantity="eta", tmax=None, plt_ax=None, color='k'):
    # calculate spin-related quantities
    # arguments
    ONEZONE = False
    num_time_chunk = 1
    zone_time_average_fraction = 0.5
    average_factor = 2
    trim_zone = True
    rescalingFactor = 1.0
    flip_sign = False
    boxcar_factor = 0
    
    with open(pkl, 'rb') as openFile:
        D = pickle.load(openFile)
        n_zones = D['nzone']
        try: n_zones_eff = D['nzone_eff']
        except: n_zones_eff = n_zones

        if quantity == 'eta':
            profiles, invert = readQuantity(D, 'Edot')
            tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
            #profiles, _ = readQuantity(D, 'Mdot')
            #_, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor, tmax)
        elif quantity == 'phib':
            profiles, invert = readQuantity(D, 'Phib')
            tDivList, usableProfiles_num, r_save, num_save = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor,tmax)
        profiles, _ = readQuantity(D, 'Mdot')
        _, usableProfiles_den, _, _ = assignTimeBins(D, profiles, ONEZONE, num_time_chunk, zone_time_average_fraction, average_factor,tmax)
      
        for b in range(num_time_chunk):
            r_plot = np.array([])
            values_plot = np.array([])
            print("{}: t={:.3g}-{:.3g}, # of run summed is {}".format(b,tDivList[b],tDivList[b+1], num_save[:,b]))
            for zone in range(n_zones_eff):
              # Now, given an average within each run, average each separate run.
                plottable_den = np.mean(usableProfiles_den[zone][b], axis=0)
                if zone == n_zones_eff-1: 
                    i10 = np.argmin(abs(r_save[zone]-10))
                    Mdot_save = plottable_den[i10]
                if quantity == 'eta':
                    plottable_num = np.mean(usableProfiles_num[zone][b], axis=0)
                    plottable = plottable_den - plottable_num #+ 4.5e-3
                elif "phib" in quantity:
                    plottable = np.mean(usableProfiles_num[zone][b], axis=0)

                # Next, optionally mask out regions that likely have wall glitches by only taking the central half of the radii
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
        
        if boxcar_factor == 0:
            boxcar_avged = values_plot[order]
        else:
            boxcar_avged = uniform_filter1d(values_plot[order], size=n_radii//boxcar_factor) # (07/12/23) boxcar averaging

        try: spin = D["spin"]
        except:
            if "_a0" in pkl: spin = float((pkl.split("/")[2])[6:].replace("_profiles_all2.pkl","")[2:])
            else: spin = 0.
        if quantity == "eta": 
            r_to_avg = (r_plot[order] > 64) & (r_plot[order] < 1e6)
            output = boxcar_avged[r_to_avg].mean()
        elif quantity == "phib": 
            iEH = np.argmin(abs(r_plot[order] - r_EH(spin)))
            output = boxcar_avged[iEH]
        if plt_ax is not None:
            lw = 2
            plt_ax.loglog(r_plot[order], boxcar_avged, color=color, lw=lw)
            if quantity == "eta": plt_ax.axhline(output, color=color, lw=lw*2, alpha=0.2, label="a={}".format(spin))
            ylabel = variableToLabel(quantity)
            plt_ax.set_ylabel(ylabel)
            xlabel = 'Radius [$r_g$]'
            plt_ax.set_xlabel(xlabel)
        return output, spin, plt_ax

def _main():
    matplotlib_settings()
    fig, axes = plt.subplots(1,3,figsize=(24,6))
    pkl_list = ["010524_a0.1", "010524_a0.3", "010524_a0.5", "011224_a0.7", "011224_a0.9"]
    colors = plt.cm.gnuplot(np.linspace(0.9,0.3,len(pkl_list)))
    phib_list = []
    spin_list = []
    for i, pkln in enumerate(pkl_list):
        pkl = "../data_products/"+pkln+"_profiles_all2.pkl"
        eta, spin, axes[0] = calc_spin_quantities(pkl, "eta", plt_ax=axes[0], color=colors[i])
        #phib, spin, axes[2] = calc_spin_quantities(pkl, "phib", plt_ax=axes[2], color=colors[i])
        phib, _, _ = calc_spin_quantities(pkl, "phib")
        phib_list += [phib]; spin_list += [spin]
        axes[1].plot(spin, eta, marker='.', color=colors[i], markersize=20)
        axes[2].plot(spin, phib, marker='.', color=colors[i], markersize=20)
    axes[0].legend()
    axes[0].set_ylim([5e-3,4])
    spin_arr = np.linspace(-1, 1, 50)
    phi = phi_fit(spin_arr) # temporary
    axes[1].plot(spin_arr, eta_BZ6(phi, spin_arr), 'b--')
    axes[1].plot(spin_list, eta_BZ6(phib_list, spin_list), 'k')
    axes[1].set_xlim([-1,1])
    axes[1].set_ylim([0,1.5])
    axes[1].set_ylabel(variableToLabel("eta"))
    axes[1].set_xlabel("a")
    axes[2].set_xlabel("a")
    axes[2].set_ylabel(variableToLabel("phib"))
    #axes[2].set_ylim([10,1e6])
    plt.savefig("./calc_eta.png")
    #pkl = "../data_products/011224_a0.9_profiles_all2.pkl"

if __name__ == "__main__":
    _main()
