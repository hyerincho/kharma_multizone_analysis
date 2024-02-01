import numpy as np
import matplotlib.pyplot as plt
import glob
import pyharm
from pyharm.plots.plot_dumps import plot_xz
import os #
import pdb
import pickle
from matplotlib_settings import *
import bondi_analytic as bondi

def t_total_to_t_atB(total_t, n_ann_out=4):
    # approximate conversion from the total runtime to runtime spent at the Bondi radius
    return total_t*2./n_ann_out # times two takes account of that 1e5 is in the overlapping region

def t_atB_to_t_total(t_atB, n_ann_out=4):
    # approximate conversion from the time spent at the Bondi radius to the total runtime
    return t_atB/2.*n_ann_out

def thphi_average(dump,quantity,sum_instead=False,mass_weight=True):
  if isinstance(quantity,str):
    to_average=np.copy(dump[quantity])
  else:
    to_average=np.copy(quantity)
  
  if mass_weight:
    to_average *= dump['rho']

  if sum_instead:
      return pyharm.shell_sum(dump,to_average)
  else:
      if mass_weight:
        return pyharm.shell_sum(dump,to_average)/pyharm.shell_sum(dump,dump["rho"])
      else:
          if dump['n3']>1: #3d
            return pyharm.shell_avg(dump,to_average)
          else:
            return np.mean(to_average,axis=1)

def phi_average(dump,quantity,sum_instead=False,mass_weight=True):
  if isinstance(quantity,str):
    to_average=np.copy(dump[quantity])
  else:
    to_average=np.copy(quantity)
  
  if mass_weight:
      #to_average *= dump['rho']
      weight = dump["rho"]
  else:
      weight = dump["1"]

  if sum_instead:
      return np.sum(to_average, axis=2)
  else:
      if dump['n3']>1:
        return np.sum(to_average * weight * np.sqrt(dump['gcov'][3,3]) * dump['dx3'], axis=2) / np.sum(weight * np.sqrt(dump['gcov'][3,3]) * dump['dx3'],axis=2)
      else:
        return to_average

def get_zone_num(dump, run_num, nzones=None):
    #try:
    #    base = float(dump["base"])
    #except:
    #    base = 8.
    #zone_num = int(np.log(dump["r_in"])/np.log(base))
    if nzones is None:
        nzones = dump["nzone_eff"]
    zone_num = (run_num % (2 * (nzones - 1)))
    if zone_num > nzones - 1: zone_num -= (nzones - 1)
    else: zone_num = (nzones - 1) - zone_num
    return zone_num

def find_edge(dirtag):
    dirs=sorted(glob.glob("../data/"+dirtag+"/*[0-9][0-9][0-9][0-9][0-9]/"))

    # find the edge thru a backward search
    for i, dr in enumerate(dirs[::-1]):
        run_num = i
        fname=sorted(glob.glob(dr+"*.rhdf"))[-1]
        dump = pyharm.load_dump(fname,ghost_zones=False)
        zone_num = get_zone_num(dump, run_num)
        if zone_num == 0 or zone_num == dump["nzone"]-1:
            edge_run = len(dirs)-1-i
            try:
                edge_iter = int(dump["iteration"])
            except:
                edge_iter = int(np.maximum(np.ceil(edge_run/(dump["nzone"]-1)),1) )
            break
    print(dirtag+" edge run: ", edge_run, " edge_iter ", edge_iter)
    return dirs, dump, edge_run, edge_iter

def get_mask(zone, n_zones, n_radii, overlap = None):
    # masking
    mask = np.full(n_radii, True, dtype=bool)
    if overlap is None: overlap = n_radii//4
    if zone < n_zones-1:
        mask[-int(overlap):] = False
    if zone > 0:
        mask[:int(overlap)] = False
    return mask

def plot_shell_summed(ax,dump,x,var,mask=None,color='k',lw=5,already_summed=False, j_slice=slice(None),avg=False, inv=False,label=None,weight=None,alpha=1):
    if inv: var = 1./var

    if not already_summed: var = np.squeeze(np.sum((var * dump['gdet'] * dump['dx2'] * dump['dx3'])[:,j_slice,:],axis=(1,2)))
    if avg: var = var/np.squeeze(np.sum((weight * dump['gdet'] * dump['dx2'] * dump['dx3'])[:,j_slice,:],axis=(1,2))) 
    
    if inv: var = 1./var

    if label is None: label="__nolegend__"
    ax.plot(x[mask], var[mask],color=color,lw=lw,label=label,alpha=alpha)
    ax.plot(x[mask], -var[mask],color=color,lw=lw, ls=':', alpha=alpha)
    return var #ax

def eta_profile(dirtag, iteration=None, boxcar_factor=0):
    from scipy.ndimage import uniform_filter1d
    matplotlib_settings()
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    try: n_zones = dump["nzone_eff"]
    except: n_zones = dump["nzone"]
    try: base = float(dump["base"])
    except: base = 8.
    r_out = np.power(base,n_zones+1)
    vmin=-1e0
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]

    FM_zones=[0]*n_zones
    Be_nob_zones=[0]*n_zones
    Edot_zones=[0]*n_zones
    Edot_Fl_zones=[0]*n_zones
    Edot_EM_zones=[0]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    eta = np.array([])
    eta_Fl = np.array([])
    eta_adv = np.array([])
    eta_conv = np.array([])
    
    # run backwards nzone times from edge_run
    if iteration is None: iteration = edge_iter//2
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump)
            FM_zones[zone] += dump["FM"]
            Be_nob_zones[zone] += dump["Be_nob"]
            Edot_zones[zone] += -pyharm.shell_sum(dump, 'FE')
            Edot_Fl_zones[zone] += -pyharm.shell_sum(dump, 'FE_Fl')
            Edot_EM_zones[zone] += -pyharm.shell_sum(dump, "FE_EM")
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    
    labels = [r'$\eta^{\rm tot}$', r'$\eta^{\rm fl}$', r'$\eta^{\rm adv}$', r'$\eta^{\rm conv}$', r'$\eta^{\rm em}$']
    window = (np.log10(2), np.log10(r_out), 0,np.pi)
    vmax=-vmin; lw = 2
    combined_r = np.array([])
    combined_eta = np.array([])
    combined_eta_fl = np.array([])
    combined_eta_em = np.array([])
    combined_eta_adv = np.array([])
    combined_eta_conv = np.array([])
    for zone in range(n_zones):
        FM_zones[zone] /= num_sum[zone]
        Be_nob_zones[zone] /= num_sum[zone]
        Edot_zones[zone] /= num_sum[zone]
        Edot_Fl_zones[zone] /= num_sum[zone]
        Edot_EM_zones[zone] /= num_sum[zone]

        # masking
        dump = dump_zones[zone]
        n_radii = len(dump["r1d"])
        mask = get_mask(zone, n_zones, n_radii, dump["nx2"]//4)

        # Mdot and Edot adv,conv calc
        Mdot = np.squeeze(np.sum(-FM_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
        Edot_adv = np.squeeze(np.sum(Be_nob_zones[zone]*FM_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
        Edot_conv = Mdot-Edot_Fl_zones[zone]-Edot_adv
        if zone == 0:
            i10 = np.argmin(abs(r_zones[zone]-10))
            Mdot_save = Mdot[i10]

        # efficiencies
        #eta = np.concatenate([eta,((Mdot-Edot_zones[zone])/Mdot_save)[mask]])
        #eta_Fl = np.concatenate([eta_Fl,((Mdot-Edot_Fl_zones[zone])/Mdot_save)[mask]])
        #eta_adv = np.concatenate([eta_adv,(Edot_adv/Mdot_save)[mask]])
        #eta_conv = np.concatenate([eta_conv,(Edot_conv/Mdot_save)[mask]])
        
        # plot
        eta_temp = (Mdot-Edot_zones[zone])/Mdot_save
        eta_fl_temp = (Mdot-Edot_Fl_zones[zone])/Mdot_save
        eta_em_temp = (-Edot_EM_zones[zone])/Mdot_save
        combined_r = np.concatenate([combined_r,np.log10(r_zones[zone])[mask]])
        combined_eta = np.concatenate([combined_eta,eta_temp[mask]])
        combined_eta_fl = np.concatenate([combined_eta_fl,eta_fl_temp[mask]])
        combined_eta_em = np.concatenate([combined_eta_em,eta_em_temp[mask]])
        combined_eta_adv = np.concatenate([combined_eta_adv,(Edot_adv/Mdot_save)[mask]])
        combined_eta_conv = np.concatenate([combined_eta_conv,(Edot_conv/Mdot_save)[mask]])
        
        #x = np.log10(r_zones[zone])
        #plot_shell_summed(ax, dump, x, (Mdot-Edot_zones[zone])/Mdot_save,      mask=mask, color='k', already_summed=True, lw=lw, label=['__nolegend__',labels[0]][(zone==0)])
        #plot_shell_summed(ax, dump, x, (Mdot-Edot_Fl_zones[zone])/Mdot_save,   mask=mask, color='b', already_summed=True, lw=lw, label=['__nolegend__',labels[1]][(zone==0)], alpha=0.7)
        #plot_shell_summed(ax, dump, x, (Edot_adv/Mdot_save),                   mask=mask, color='g', already_summed=True, lw=lw, label=['__nolegend__',labels[2]][(zone==0)], alpha=0.7)
        #plot_shell_summed(ax, dump, x, (Edot_conv/Mdot_save),                  mask=mask, color='r', already_summed=True, lw=lw, label=['__nolegend__',labels[3]][(zone==0)], alpha=0.7)
 
    colors = ['k', 'b', 'g', 'r', 'c']
    for i,combined in enumerate([combined_eta, combined_eta_fl, combined_eta_adv, combined_eta_conv, combined_eta_em]):
        if boxcar_factor > 0: combined = uniform_filter1d(combined, size=dump["nx2"]//boxcar_factor) # (07/12/23) boxcar averaging
        ax.plot(combined_r,combined,color=colors[i],lw=lw,label=labels[i])
        ax.plot(combined_r,-combined,color=colors[i],ls=':',lw=lw)
    
    ax.set_yscale('log'); ax.set_ylim([vmax/1e4,vmax]); ax.legend(fontsize=18) #,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.8))
    r_sonic = dump["rs"]
    rB = 80.*r_sonic**2/(27.*dump['gam'])
    ax.axvline(np.log10(rB), color='grey', lw=2, alpha=1) # show R_B
    ax.set_xlabel(r'$\log_{10}(r)$'); ax.set_xlim([np.log10(2),np.log10(r_zones[n_zones-1][-1])]); ax.set_ylabel(r'$\eta$')
    ax.axhline(0.01,color='m', lw=3, alpha=0.2) # horizontal line to show 2%
    fig.tight_layout()
    plt.savefig("../plots/eta_profile.pdf",bbox_inches='tight')
    plt.close(fig)

def Omega_slice(dirtag, iteration=None, avg=True):
    matplotlib_settings()
    fig, ax = plt.subplots(2,2,figsize=(12,6),sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    try: n_zones = dump["nzone_eff"]
    except: n_zones = dump["nzone"]
    try: base = float(dump["base"])
    except: base = 8.
    r_out = np.power(base,n_zones+1)
    vmin=-1e0
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]

    rho_zones=[0]*n_zones
    Omega_zones=[0]*n_zones
    abs_Omega_zones=[0]*n_zones
    Omega_r_zones=[0]*n_zones
    abs_Omega_r_zones=[0]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    
    # run backwards nzone times from edge_run
    if iteration is None: iteration = edge_iter//2  # 100 #
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump)
            rho_zones[zone] += phi_average(dump,"rho",mass_weight=0)
            Omega_zones[zone] += phi_average(dump,"Omega")
            abs_Omega_zones[zone] += phi_average(dump,"abs_Omega")
            Omega_r_zones[zone] += thphi_average(dump,"Omega")
            abs_Omega_r_zones[zone] += thphi_average(dump,"abs_Omega")
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    window = (np.log10(2), np.log10(r_out), 0,np.pi)
    vmax=-vmin; lw = 2
    for zone in range(n_zones):
        rho_zones[zone] /= num_sum[zone]
        Omega_zones[zone] /= num_sum[zone]
        abs_Omega_zones[zone] /= num_sum[zone]
        Omega_r_zones[zone] /= num_sum[zone]
        abs_Omega_r_zones[zone] /= num_sum[zone]

        # normalize with Keplerian
        Omega_zones[zone] *= np.power(r_zones[zone][:,np.newaxis],3./2)
        abs_Omega_zones[zone] *= np.power(r_zones[zone][:,np.newaxis],3./2)
        Omega_r_zones[zone] *= np.power(r_zones[zone],3./2)
        abs_Omega_r_zones[zone] *= np.power(r_zones[zone],3./2)

        # masking
        dump = dump_zones[zone]
        n_radii = len(dump["r1d"])
        mask = get_mask(zone, n_zones, n_radii, dump["nx2"]//4)

        # plot 
        x = np.log10(r_zones[zone])
        plot_shell_summed(ax[0,0],dump,x,Omega_r_zones[zone],mask=mask, color='k', lw=lw, already_summed=1)
        plot_shell_summed(ax[0,1],dump,x,abs_Omega_r_zones[zone],mask=mask, color='k', lw=lw, already_summed=1)
        ax[0,0].plot(x[mask],Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',lw=lw)
        ax[0,0].plot(x[mask],-Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',ls=':',lw=lw)
        ax[0,1].plot(x[mask],abs_Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',lw=lw)
        plot_xz(ax[1,0], dump, Omega_zones[zone]    , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=(zone==0),shading='flat',window=window,average=0, mask=mask)#/rho_zones[zone]
        plot_xz(ax[1,1], dump, abs_Omega_zones[zone], native=True, log=True, vmin=1e-3,vmax=vmax, cbar=(zone==0),shading='flat',window=window,average=0, mask=mask) #/rho_zones[zone]

    
    for j in range(2): ax[0,j].set_yscale('log'); ax[0,j].set_ylim([1e-3,vmax])#; ax[0,j].legend(fontsize=15,ncol=2)
    ax[0,0].set_title(r'$\langle \Omega\rangle/\Omega_K$')
    ax[0,1].set_title(r'$\langle |\Omega|\rangle/\Omega_K$')
    
    fig.tight_layout()
    plt.savefig("../plots/Omega_slice.pdf",bbox_inches='tight')
    plt.close(fig)

def FE_slice(dirtag, iteration=None, avg=True, r_slice=None):
    matplotlib_settings()
    if r_slice is None: fig, ax = plt.subplots(2,3,figsize=(16,6),sharex=True, sharey='row', gridspec_kw={'height_ratios': [1, 2]})
    else: fig, ax = plt.subplots(2,3,figsize=(16,6),sharex=False, gridspec_kw={'height_ratios': [1, 2]})
    plt.subplots_adjust(hspace=0.4, wspace=0.01)
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    try: n_zones = dump["nzone_eff"]
    except: n_zones = dump["nzone"]
    try: base = float(dump["base"])
    except: base = 8.
    r_out = np.power(base,dump["nzone"]+1)
    if r_slice is None: 
        if dump["a"] > 0: vmin=-4 #-5e-1#
        else: vmin=-1e-1 #-1e-2 #
    else: vmin=-5e-2#-1e-3
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]

    FM_zones=[0]*n_zones
    FE_zones=[0]*n_zones
    FE_Fl_zones=[0]*n_zones
    FE_EM_zones=[0]*n_zones
    FE_adv_zones=[0]*n_zones
    FE_conv_zones=[0]*n_zones
    Be_nob_zones=[0]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    th_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    
    # run backwards nzone times from edge_run
    if iteration is None: iteration = edge_iter//2  # 100 #
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
        #for file_ in files[int(len(files)*(1.-1./np.sqrt(2))):]:  # only add last sqrt(2)
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump, edge_run-i)
            FM_zones[zone] += dump["FM"]
            FE_zones[zone] += dump["FE_norho"]
            FE_Fl_zones[zone] += dump["FE_Fl"]-dump["rho"]*dump['ucon'][1]
            FE_EM_zones[zone] += dump["FE_EM"]
            Be_nob_zones[zone] += dump["Be_nob"]
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if th_zones[zone] is None:
                th_zones[zone] = dump["th1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    #labels = [r'$\dot{E}_{\rm net}$', r'$\dot{E}^{\rm fl}_{\rm net}$', r'$\dot{E}^{\rm adv}_{\rm net}$', r'$\dot{E}^{\rm conv}_{\rm net}$']
    labels = [r'$\eta^{\rm tot}$', r'$\eta^{\rm fl}$', r'$\eta^{\rm adv}$', r'$\eta^{\rm conv}$', r'$\eta^{\rm em}$']
    window = (np.log10(2), np.log10(r_out), 0,np.pi)
    vmax=-vmin; lw = 2
    for zone in range(n_zones):
        FE_zones[zone] /= num_sum[zone]
        FE_Fl_zones[zone] /= num_sum[zone]
        FE_EM_zones[zone] /= num_sum[zone]
        Be_nob_zones[zone] /= num_sum[zone]
        FM_zones[zone] /= num_sum[zone]
        FE_adv_zones[zone] = Be_nob_zones[zone]*FM_zones[zone] # advection
        FE_conv_zones[zone] = FE_Fl_zones[zone]-FE_adv_zones[zone] # convection

        # masking: TODO: check if this is applied for moving_rin
        dump = dump_zones[zone]
        n_radii = len(dump["r1d"])
        mask = get_mask(zone, n_zones, n_radii, dump["nx2"]//4)

        # plot 
        Edot_net = np.squeeze(np.sum(FE_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
        Mdot = np.squeeze(np.sum(-FM_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
        if zone == 0:
            i10 = np.argmin(abs(r_zones[zone]-10))
            Mdot_save = Mdot[i10]
        x = np.log10(r_zones[zone])
        if r_slice is None:
            for i,ax_each in enumerate(ax[0,:]): plot_shell_summed(ax_each,dump,x,Edot_net/Mdot_save,mask=mask, color='k',already_summed=True, lw=lw, label=['__nolegend__',labels[0]][(zone==0) and (i==0)])
            plot_shell_summed(ax[0,0],dump,x,FE_Fl_zones[zone]/Mdot_save,mask=mask, color='b', lw=lw, label=['__nolegend__',labels[1]][(zone==0)])
            adv = plot_shell_summed(ax[0,1], dump,x,FE_adv_zones[zone]/Mdot_save,mask=mask, color='g', lw=lw, label=['__nolegend__',labels[2]][(zone==0)])
            conv = plot_shell_summed(ax[0,2],dump,x,FE_conv_zones[zone]/Mdot_save,mask=mask, color='r', lw=lw, label=['__nolegend__',labels[3]][(zone==0)])
            em = plot_shell_summed(ax[0,0],dump,x,FE_EM_zones[zone]/Mdot_save,mask=mask, color='c', lw=lw, label=['__nolegend__',labels[4]][(zone==0)])
        else:
            colors=plt.cm.gnuplot(np.linspace(0.9,0.3,len(r_slice)))
            for j,r_chosen in enumerate(r_slice):
                if r_chosen<max(r_zones[zone][mask]) and r_chosen> min(r_zones[zone][mask]):
                    print(zone)
                    idx = np.argmin(abs(r_zones[zone]-r_chosen))
                    FE_Fl_phiav = (FE_Fl_zones[zone]*dump["gdet"]/Mdot_save).sum(-1)/dump['n3']
                    FE_adv_phiav = (FE_adv_zones[zone]*dump["gdet"]/Mdot_save).sum(-1)/dump['n3']
                    FE_conv_phiav = (FE_conv_zones[zone]*dump["gdet"]/Mdot_save).sum(-1)/dump['n3']
                    ax[0,0].plot(th_zones[zone], FE_Fl_phiav[idx],label=r'{}$r_g$'.format(r_chosen), color=colors[j])
                    ax[0,0].plot(th_zones[zone], -FE_Fl_phiav[idx],':', color=colors[j])
                    ax[0,1].plot(th_zones[zone], FE_adv_phiav[idx], color=colors[j])
                    ax[0,1].plot(th_zones[zone], -FE_adv_phiav[idx],':', color=colors[j])
                    ax[0,2].plot(th_zones[zone], FE_conv_phiav[idx], color=colors[j])
                    ax[0,2].plot(th_zones[zone], -FE_conv_phiav[idx],':', color=colors[j])
        plot_xz(ax[1,0], dump, FE_Fl_zones[zone]  * dump["gdet"]/Mdot_save , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        plot_xz(ax[1,1], dump, FE_adv_zones[zone] * dump["gdet"]/Mdot_save , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        im=plot_xz(ax[1,2], dump, FE_conv_zones[zone]* dump["gdet"]/Mdot_save , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=0,shading='flat',window=window,average=avg, mask=mask)
        if zone == 0:
            cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.45])
            fig.colorbar(im, cax = cbar_ax,ticks=[-0.1,-0.01,-0.001,-0.0001,0,0.0001,0.001,0.01,0.1])#ax=ax[1,2], pad=0.5)

    
    r_sonic = dump["rs"]
    gam=5./3.
    rB = 80.*r_sonic**2/(27.*gam)
    for j in range(3): ax[0,j].set_yscale('log'); ax[0,j].set_ylim([vmax/1e2,vmax]); 
    if r_slice is None:
        #for j in range(3): ax[0,j].legend(fontsize=18,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.8))
        lines_labels = [ax.get_legend_handles_labels() for ax in ax[0,:]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        #fig.legend(lines, labels, ncol=1, fontsize=14, bbox_to_anchor=(0.99,0.95))
        fig.legend(lines, labels, ncol=5, bbox_to_anchor=(0.7,1.01))
        for ax_each in ax.reshape(-1): ax_each.axvline(np.log10(rB), color='grey', lw=2, alpha=1,ls="--") # show R_B
    else:
        ax[0,0].legend(fontsize=18,ncol=2,loc='upper center', bbox_to_anchor=(0.5, 1.8))
        for j in range(3): ax[0,j].set_xlabel(r'$\theta$')
    ax[1,1].set_ylabel('')
    ax[1,2].set_ylabel('')
    ax[1,0].set_title(r'Fluid $F_E^{fl}/\overline{\dot{M}}_{10}$')
    ax[1,1].set_title(r'Advection $F_E^{adv}/\overline{\dot{M}}_{10}$')
    ax[1,2].set_title(r'Convection $F_E^{conv}/\overline{\dot{M}}_{10}$')
    dump = pyharm.load_dump(glob.glob(dirs[edge_run]+"/*final.phdf")[0],ghost_zones=False)
    if 1: ax[0,2].set_title('last #{} t = {:.3g}'.format(edge_run,dump["t"]),fontsize=13)
    #fig.tight_layout()
    save_path="../plots/runs/"+dirtag
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    savefig_name = "/FE_slice.png"
    plt.savefig(save_path+savefig_name,bbox_inches='tight')
    plt.savefig("../plots/"+savefig_name,bbox_inches='tight')
    plt.close(fig)

def FE_compare(dirtags, iteration=None, avg=True, r_slice=1e4, colors=['k','b'], labels=None):
    matplotlib_settings()
    fig, ax = plt.subplots(1,3,figsize=(16,4),sharex=True)
    
    if labels is None: labels=['__nolegend__']*len(dirtags)
    
    for k,dirtag in enumerate(dirtags):
        dirs, dump, edge_run, edge_iter = find_edge(dirtag)
        
        try: n_zones = dump["nzone_eff"]
        except: n_zones = dump["nzone"]
        try: base = float(dump["base"])
        except: base = 8.
        r_out = np.power(base,n_zones+1)
        vmin=-1e-2#-1e-3
        if "onezone" in dirtag or "oz" in dirtag:
            n_zones = 1
            r_out = dump["r_out"]

        FM_zones=[0]*n_zones
        FE_zones=[0]*n_zones
        FE_Fl_zones=[0]*n_zones
        FE_adv_zones=[0]*n_zones
        FE_conv_zones=[0]*n_zones
        Be_nob_zones=[0]*n_zones
        num_sum=[0]*n_zones
        r_zones=[None]*n_zones
        th_zones=[None]*n_zones
        dump_zones=[None]*n_zones
        
        # run backwards nzone times from edge_run
        if iteration is None: iteration = edge_iter//2  # 100 #
        for i in range(iteration*(n_zones-1)+1):
            files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
            for file_ in files[len(files)//2:]:  # only add last half
                dump = pyharm.load_dump(file_,ghost_zones=False)
                zone = get_zone_num(dump)
                FM_zones[zone] += dump["FM"]
                FE_zones[zone] += dump["FE_norho"]
                FE_Fl_zones[zone] += dump["FE_Fl"]-dump["rho"]*dump['ucon'][1]
                Be_nob_zones[zone] += dump["Be_nob"]
                num_sum[zone]+=1
                if r_zones[zone] is None:
                    r_zones[zone] = dump["r1d"]
                if th_zones[zone] is None:
                    th_zones[zone] = dump["th1d"]
                if dump_zones[zone] is None:
                    dump_zones[zone] = dump
        print(num_sum)
        window = (np.log10(2), np.log10(r_out), 0,np.pi)
        vmax=-vmin; lw = 2
        for zone in range(n_zones):
            FE_zones[zone] /= num_sum[zone]
            FE_Fl_zones[zone] /= num_sum[zone]
            Be_nob_zones[zone] /= num_sum[zone]
            FM_zones[zone] /= num_sum[zone]
            FE_adv_zones[zone] = Be_nob_zones[zone]*FM_zones[zone] # advection
            FE_conv_zones[zone] = FE_Fl_zones[zone]-FE_adv_zones[zone] # convection

            # masking
            dump = dump_zones[zone]
            n_radii = len(dump["r1d"])
            mask = get_mask(zone, n_zones, n_radii, dump["nx2"]//4)

            # plot 
            Edot_net = np.squeeze(np.sum(FE_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
            Mdot = np.squeeze(np.sum(-FM_zones[zone] * dump['gdet'] * dump['dx2'] * dump['dx3'],axis=(1,2)))
            if zone == 0:
                i10 = np.argmin(abs(r_zones[zone]-10))
                Mdot_save = Mdot[i10]
            x = np.log10(r_zones[zone])
            if r_slice<max(r_zones[zone][mask]) and r_slice> min(r_zones[zone][mask]):
                print(zone)
                idx = np.argmin(abs(r_zones[zone]-r_slice))
                FE_Fl_phiav = (FE_Fl_zones[zone]*dump["gdet"]/Mdot_save).sum(-1)/dump['n3']
                FE_adv_phiav = (FE_adv_zones[zone]*dump["gdet"]/Mdot_save).sum(-1)/dump['n3']
                FE_conv_phiav = (FE_conv_zones[zone]*dump["gdet"]/Mdot_save).sum(-1)/dump['n3']
                ax[0].plot(th_zones[zone], FE_Fl_phiav[idx], color=colors[k], label=labels[k])#, ls=linestyles[k])
                ax[0].plot(th_zones[zone], -FE_Fl_phiav[idx],':', color=colors[k])
                ax[1].plot(th_zones[zone], FE_adv_phiav[idx], color=colors[k])#, ls=linestyles[k])
                ax[1].plot(th_zones[zone], -FE_adv_phiav[idx],':', color=colors[k])
                ax[2].plot(th_zones[zone], FE_conv_phiav[idx], color=colors[k])#, ls=linestyles[k])
                ax[2].plot(th_zones[zone], -FE_conv_phiav[idx],':', color=colors[k])
        
    for j in range(3): 
        ax[j].set_yscale('log'); ax[j].set_ylim([vmax/1e2,vmax]); ax[j].set_xlabel(r'$\theta$')
    ax[0].set_title('Fluid')
    ax[1].set_title('Advection')
    ax[2].set_title('Convection')
    ax[0].legend(fontsize=18,ncol=2,loc='upper center')#, bbox_to_anchor=(0.5, 1.8))
    fig.suptitle(r'$r={:.3g} rg$'.format(r_slice))
    fig.tight_layout()
    plt.savefig("../plots/FE_compare.png",bbox_inches='tight')
    plt.close(fig)

def Omega_slice(dirtag,avg=True):
    matplotlib_settings()
    fig, ax = plt.subplots(2,2,figsize=(12,6),sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    try: n_zones = dump["nzone_eff"]
    except: n_zones = dump["nzone"]
    try: base = float(dump["base"])
    except: base = 8.
    r_out = np.power(base,n_zones+1)
    vmin=-1e0
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]

    rho_zones=[0]*n_zones
    Omega_zones=[0]*n_zones
    abs_Omega_zones=[0]*n_zones
    Omega_r_zones=[0]*n_zones
    abs_Omega_r_zones=[0]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    
    # run backwards nzone times from edge_run
    iteration = edge_iter//2  # 100 #
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump)
            rho_zones[zone] += phi_average(dump,"rho",mass_weight=0)
            Omega_zones[zone] += phi_average(dump,"Omega")
            abs_Omega_zones[zone] += phi_average(dump,"abs_Omega")
            Omega_r_zones[zone] += thphi_average(dump,"Omega")
            abs_Omega_r_zones[zone] += thphi_average(dump,"abs_Omega")
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    window = (np.log10(2), np.log10(r_out), 0,np.pi)
    vmax=-vmin; lw = 2
    for zone in range(n_zones):
        rho_zones[zone] /= num_sum[zone]
        Omega_zones[zone] /= num_sum[zone]
        abs_Omega_zones[zone] /= num_sum[zone]
        Omega_r_zones[zone] /= num_sum[zone]
        abs_Omega_r_zones[zone] /= num_sum[zone]

        # normalize with Keplerian
        Omega_zones[zone] *= np.power(r_zones[zone][:,np.newaxis],3./2)
        abs_Omega_zones[zone] *= np.power(r_zones[zone][:,np.newaxis],3./2)
        Omega_r_zones[zone] *= np.power(r_zones[zone],3./2)
        abs_Omega_r_zones[zone] *= np.power(r_zones[zone],3./2)

        # masking
        dump = dump_zones[zone]
        n_radii = len(dump["r1d"])
        mask = get_mask(zone, n_zones, n_radii, dump["nx2"]//4)

        # plot 
        x = np.log10(r_zones[zone])
        plot_shell_summed(ax[0,0],dump,x,Omega_r_zones[zone],mask=mask, color='k', lw=lw, already_summed=1)
        plot_shell_summed(ax[0,1],dump,x,abs_Omega_r_zones[zone],mask=mask, color='k', lw=lw, already_summed=1)
        ax[0,0].plot(x[mask],Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',lw=lw)
        ax[0,0].plot(x[mask],-Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',ls=':',lw=lw)
        ax[0,1].plot(x[mask],abs_Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',lw=lw)
        plot_xz(ax[1,0], dump, Omega_zones[zone]    , native=True, symlog=True, vmin=vmin,vmax=vmax, cbar=(zone==0),shading='flat',window=window,average=0, mask=mask)#/rho_zones[zone]
        plot_xz(ax[1,1], dump, abs_Omega_zones[zone], native=True, log=True, vmin=1e-3,vmax=vmax, cbar=(zone==0),shading='flat',window=window,average=0, mask=mask) #/rho_zones[zone]

    
    for j in range(2): ax[0,j].set_yscale('log'); ax[0,j].set_ylim([1e-3,vmax])#; ax[0,j].legend(fontsize=15,ncol=2)
    ax[0,0].set_title(r'$\langle \Omega\rangle/\Omega_K$')
    ax[0,1].set_title(r'$\langle |\Omega|\rangle/\Omega_K$')
    
    fig.tight_layout()
    plt.savefig("../plots/Omega_slice.pdf",bbox_inches='tight')
    plt.close(fig)

def Omega_profile(dirtag, iteration=None, avg=True):
    matplotlib_settings()
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    
    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    try: n_zones = dump["nzone_eff"]
    except: n_zones = dump["nzone"]
    try: base = float(dump["base"])
    except: base = 8.
    r_out = np.power(base,n_zones+1)
    vmin=-1e0
    if "onezone" in dirtag or "oz" in dirtag:
        n_zones = 1
        r_out = dump["r_out"]

    Omega_zones=[0]*n_zones
    abs_Omega_zones=[0]*n_zones
    num_sum=[0]*n_zones
    r_zones=[None]*n_zones
    dump_zones=[None]*n_zones
    
    # run backwards nzone times from edge_run
    if iteration is None: iteration = edge_iter//2  # 100 #
    for i in range(iteration*(n_zones-1)+1):
        files=sorted(glob.glob(dirs[edge_run-i]+"/*.phdf")) # HYERIN TEST rhdf -> phdf
        for file_ in files[len(files)//2:]:  # only add last half
            dump = pyharm.load_dump(file_,ghost_zones=False)
            zone = get_zone_num(dump)
            Omega_zones[zone] += phi_average(dump,"Omega")
            abs_Omega_zones[zone] += phi_average(dump,"abs_Omega")
            num_sum[zone]+=1
            if r_zones[zone] is None:
                r_zones[zone] = dump["r1d"]
            if dump_zones[zone] is None:
                dump_zones[zone] = dump
    print(num_sum)
    window = (np.log10(2), np.log10(r_out), 0,np.pi)
    vmax=-vmin; lw = 2
    combined_r = np.array([])
    combined_O = np.array([])
    for zone in range(n_zones):
        Omega_zones[zone] /= num_sum[zone]
        abs_Omega_zones[zone] /= num_sum[zone]

        # normalize with Keplerian
        Omega_zones[zone] *= np.power(r_zones[zone][:,np.newaxis],3./2)
        abs_Omega_zones[zone] *= np.power(r_zones[zone][:,np.newaxis],3./2)

        # masking
        dump = dump_zones[zone]
        n_radii = len(dump["r1d"])
        mask = get_mask(zone, n_zones, n_radii, dump["nx2"]//4)

        # plot 
        x = np.log10(r_zones[zone])
        #combined_r = np.concatenate([combined_r,np.log10(r_zones[zone])[mask]])
        #combined_O = np.concatenate([combined_O,Omega_zones[zone][:,dump["nx2"]//2][mask]])
        ax.plot(x[mask],Omega_zones[zone][:,dump["nx2"]//2][mask],color='k',lw=lw,label=['__nolegend__',r'$\langle\Omega_{\rm mid}\rangle/\Omega_K$'][(zone==0)])
        ax.plot(x[mask],-Omega_zones[zone][:,dump["nx2"]//2][mask],color='k',ls=':',lw=lw)
        ax.plot(x[mask],abs_Omega_zones[zone][:,dump["nx2"]//2][mask],color='b',lw=lw,label=['__nolegend__',r'$\langle|\Omega_{\rm mid}|\rangle/\Omega_K$'][(zone==0)])
    #ax.plot(combined_r,combined_O,color='k',lw=lw,label=r'$\langle\Omega_{\rm mid}\rangle/\Omega_K$')
    #ax.plot(combined_r,-combined_O,color='k',ls=':',lw=lw)

    
    ax.set_yscale('log'); ax.set_ylim([1e-3,vmax]); ax.legend(fontsize=15,ncol=2,loc='upper center')
    ax.set_xlabel(r'$\log_{10}(r)$'); ax.set_xlim([np.log10(2),np.log10(r_zones[n_zones-1][-1])]);
    
    fig.tight_layout()
    plt.savefig("../plots/Omega_profile.pdf",bbox_inches='tight')
    plt.close(fig)

def plot_snapshot(dirtag,show_mdot_phib=True):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from pyharm.plots.overlays import overlay_field, overlay_streamlines_xz
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    matplotlib_settings()
    plt.rcParams.update({'font.size': 25})

    #fig = plt.figure(figsize=(24,14))

    if show_mdot_phib:
        fig = plt.figure(figsize=(25,15)) #16
        outer = GridSpec(2, 1, height_ratios = [1, 5])  # 6
        # mdot_phib vs t plot
        ax_t = plt.subplot(outer[0])
        ax_t.xaxis.tick_top()
        ax_t.xaxis.set_label_position('top') 
        pkl = '../data_products/'+dirtag+'_profiles_all2.pkl'
        ax_t = mdot_phib_t(pkl,ax_passed=ax_t)
        spec_snp = outer[1]
    else:
        fig = plt.figure(figsize=(25,13))
        spec_snp = GridSpec(1, 1)[0]
    plt.subplots_adjust(hspace=0.01)#, left=0, right=0.93, bottom=0, top=0.95)

    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec = spec_snp, hspace=0.02, wspace=0.02)
    ax=[]
    for cell in gs: ax += [plt.subplot(cell)]
    ax = np.array(ax).reshape(2,-1)
    #gs = GridSpec(5, 4, figure=fig)
    #axes = [fig.add_subplot(gs[0, :])]
    #axes += [fig.add_subplot(gs[1:3,i]) for i in range(4)]
    #axes += [fig.add_subplot(gs[3:5,3-i]) for i in range(4)]

    dirs, dump, edge_run, edge_iter = find_edge(dirtag)
    
    try: n_zones = dump["nzone_eff"]
    except: n_zones = dump["nzone"]

    plotrc={}
    plotrc.update({'xlabel': False, 'ylabel': False,'xticks': [], 'yticks': [],'cbar': False, 'frame': False, 'no_title': True, 'shading': 'flat'})
    patch_rout = np.zeros(n_zones)
    f = 1/np.sqrt(2)

    axes = np.concatenate((ax[0],ax[1,::-1]))
    #edge_run -= 7 # wind back
    for i, ax1d in enumerate(axes):
        fn=glob.glob(dirs[edge_run-i]+"/*final.rhdf")[0]
        dump = pyharm.load_dump(fn,ghost_zones=False)
        r_out = dump["r_out"]
        window = (-r_out*f, r_out*f, -r_out*f, r_out*f)
        im1 = plot_xz(ax1d, dump, "log_beta", window=window, cmap='plasma', vmin=1e-1, vmax=1e3, **plotrc)
        im2 = plot_xz(ax1d, dump, dump["rho"], window=window, half_cut=True, vmin=1e-10, vmax=1e-4, log=True, **plotrc) #*np.sqrt(dump["r"]) # 1e-5,1e-3
        scale = np.power(10,np.floor(np.log10(r_out)))
        #if r_out < 1e3: c='k'
        #else: 
        c= 'white'
        #if i>=1 and i<=3:
        #    c='k'
        try: base = float(dump["base"])
        except: base = 8.
        scalebar = AnchoredSizeBar(ax1d.transData, scale, r'$10^{:d}\, r_g$'.format(int(np.log10(scale))), 'lower left', pad=0.5, color=c, frameon=False, size_vertical=dump["r_out"]/base**2)
        ax1d.add_artist(scalebar)
        overlay_field(ax1d, dump, half_cut=True,nlines=10, reverse=True)#, sum=False)
        #overlay_streamlines_xz(ax1d, dump, 'B1', 'B2', native=False, half_cut=True)
        ax1d.title.set_visible(False)
	
        if i ==0:
            if dump["r_in"] <2: inwards=-1
            else: inwards=1
            ax1d.text(-r_out*f*0.15, r_out*f*0.8, r'$\beta$', color='w') # 0.8
            ax1d.text(r_out*f*0.05, r_out*f*0.8, r'$\rho$', color='w') # 'k'
        
        if (i==0):# (inwards>0 and i==0) or (inwards<0 and i == n_zones-1):
            # colorbar only at the outermost annulus
            #cb1=fig.colorbar(im1, cax=ax1d.inset_axes((0.01, 0.25, 0.05, 0.5))) # for beta
            cb1=fig.colorbar(im1, cax=ax1d.inset_axes((-0.1, 0.15, 0.05, 0.7))) # for beta
            #cb2=fig.colorbar(im2, cax=ax1d.inset_axes((0.51, 0.25, 0.05, 0.5))) # for rho
            #cb1.yaxis.set_ticks_position('left')
            cb1.ax.tick_params(labelleft=True, labelright=False)
            #for cb in [cb1]: #,cb2]:
                # set colorbar tick color
                #cb.ax.yaxis.set_tick_params(color='w', labelcolor='w')
                # set colorbar edgecolor 
                #cb.outline.set_edgecolor('w')
        if i==3:
            cb2=fig.colorbar(im2, cax=ax1d.inset_axes((1.02, 0.15, 0.05, 0.7))) # for rho
        
        patch_rout[i] = r_out
    
    for i, ax1d in enumerate(axes):
        if i+inwards < len(axes) and i+inwards>=0:
            rect = patches.Rectangle((-patch_rout[i+inwards]*f,-patch_rout[i+inwards]*f), 2*patch_rout[i+inwards]*f, 2*patch_rout[i+inwards]*f, linewidth=3, edgecolor='w', facecolor='none')
            ax1d.add_patch(rect)
            if i < 4 - (inwards > 0):
                con1 = patches.ConnectionPatch(xyA=(inwards*patch_rout[i+inwards]*f, -patch_rout[i+inwards]*f), xyB=(-inwards*patch_rout[i+inwards]*f,-patch_rout[i+inwards]*f), coordsA="data", coordsB="data", axesA=ax1d, axesB=axes[i+inwards], color='w',ls=':', lw=2)
                con2 = patches.ConnectionPatch(xyA=(inwards*patch_rout[i+inwards]*f, patch_rout[i+inwards]*f), xyB=(-inwards*patch_rout[i+inwards]*f,patch_rout[i+inwards]*f), coordsA="data", coordsB="data", axesA=ax1d, axesB=axes[i+inwards], color='w',ls=':', lw=2)
            if i == 4 - (inwards > 0):
                con1 = patches.ConnectionPatch(xyA=(patch_rout[i+inwards]*f, -inwards*patch_rout[i+inwards]*f), xyB=(patch_rout[i+inwards]*f,inwards*patch_rout[i+inwards]*f), coordsA="data", coordsB="data", axesA=ax1d, axesB=axes[i+inwards], color='w',ls=':', lw=2)
                con2 = patches.ConnectionPatch(xyA=(-patch_rout[i+inwards]*f, -inwards*patch_rout[i+inwards]*f), xyB=(-patch_rout[i+inwards]*f,inwards*patch_rout[i+inwards]*f), coordsA="data", coordsB="data", axesA=ax1d, axesB=axes[i+inwards], color='w',ls=':', lw=2)
            if i > 4 - (inwards > 0):
                con1 = patches.ConnectionPatch(xyA=(-inwards*patch_rout[i+inwards]*f, -patch_rout[i+inwards]*f), xyB=(inwards*patch_rout[i+inwards]*f,-patch_rout[i+inwards]*f), coordsA="data", coordsB="data", axesA=ax1d, axesB=axes[i+inwards], color='w',ls=':', lw=2)
                con2 = patches.ConnectionPatch(xyA=(-inwards*patch_rout[i+inwards]*f, patch_rout[i+inwards]*f), xyB=(inwards*patch_rout[i+inwards]*f,patch_rout[i+inwards]*f), coordsA="data", coordsB="data", axesA=ax1d, axesB=axes[i+inwards], color='w',ls=':', lw=2)

            fig.add_artist(con1)
            fig.add_artist(con2)
        
        # panel numbers
        ax1d.text(0.02, 0.93, 'zone-'+str([0,7][inwards>0]-i*(inwards)),transform=ax1d.transAxes, fontsize=25, color='w')#, bbox=dict(facecolor='w', edgecolor='k', pad=5.0))
        
    plt.savefig("../plots/snapshot.png",bbox_inches='tight')
    plt.close()

def mdot_phib_t(pkl,ax_passed=None):
    from plotProfiles import readQuantity, assignTimeBins

    matplotlib_settings()
    plt.rcParams.update({'font.size': 25})
    if ax_passed is None:
        fig, ax = plt.subplots(1,1,figsize=(24,3))
    else:
        ax = ax_passed

    t_list = np.array([])
    Phib = np.array([])
    Mdot = np.array([])
    Mdot_save = np.array([])
    
    with open(pkl, 'rb') as openFile:
        D = pickle.load(openFile)
        
        try: n_zones = D['nzone_eff']
        except : n_zones = D['nzone']
        radii = D['radii']
        try: base = D['base']
        except: base = 8
        rB = 80.*D["r_sonic"]**2/(27.*5/3)
        tB = np.power(rB,3./2)
        rescale_value = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs=np.sqrt(1e5))[0]
        
        # number of annuli that has its r_out > rB
        n_ann_out = 0
        for i in range(n_zones):
            if radii[i][-1]>rB:
                n_ann_out += 1
            
        profiles, _ = readQuantity(D, 'Phib')
        for i,profile in enumerate(profiles):
            zone_num = int(np.floor(np.log(radii[i][0])/np.log(base)))
            if zone_num == 0: # innermost zone
                r_innermost = radii[i]
                iEH = np.argmin(abs(r_innermost-2))
                times = np.array(D["times"][i])
                #if len(t_list)<1: t_list = times-times[0]
                #else: t_list = np.concatenate((t_list, times-times[0]+t_list[-1]))
                t_list = np.concatenate((t_list,t_total_to_t_atB(times,n_ann_out)/tB))
                Phib = np.concatenate((Phib,np.array(profile)[:,iEH]))
        profiles, _ = readQuantity(D, 'Mdot')
        for i,profile in enumerate(profiles):
            zone_num = int(np.floor(np.log(radii[i][0])/np.log(base)))
            if zone_num == 0: # innermost zone
                Mdot = np.concatenate((Mdot,np.array(profile)[:,iEH]))
            if zone_num == 1: # next innermost zone
                i10 = np.argmin(abs(radii[i]-10))
                #t_list2 = np.concatenate((t_list2,t_total_to_t_atB(np.array(D["times"][i]),n_ann_out)/tB))
                Mdot_save = np.concatenate((Mdot_save,np.array(profile)[:,i10]))

        Mdot_save = np.mean(Mdot_save[int(len(Mdot_save)/2.):]) # TODO: change this to time criterion by getting indices over t_half
        phib = Phib/np.sqrt(Mdot_save)
    
    ax2 = ax.twinx()
    #t_list /=  np.power(64,3./2) # free-fall time at r_out of the smallest annulus
    ax.semilogy(t_list,Mdot/rescale_value, color='k')#,marker='.')
    ax.axhline(Mdot_save/rescale_value,color ='grey', lw=3, alpha=0.5)
    ax2.semilogy(t_list,phib, color='b')#,marker='.')
    mean_phib = np.mean(phib[len(phib)//2:])
    print('mean phib: {:.3g} and mean M10: {:.3g}'.format(mean_phib, Mdot_save))
    ax2.axhline(mean_phib,color ='b',lw=3, alpha=0.3)
    ax.set_ylim([1e-4,1])
    ax2.set_ylim([1e-1,1e2])
    ax2.tick_params(axis='y', labelcolor='b')
    ax.set_ylabel(r'$\dot{M}(r=2)$ $[\dot{M}_B]$')
    ax2.set_ylabel(r'$\phi_b(r=2)$', color='b')
    ax.set_xlabel(r'$t_{\rm run}(R_B)$ $[t_B]$', labelpad=10)
    ax.set_xlim([t_list[0], t_list[-1]])#; ax.set_xscale('log')
    ax.axvspan(t_list[-1]/2., t_list[-1], facecolor='pink', alpha=0.5)

    
    if ax_passed is None:
        fig.tight_layout()
        plt.savefig("../plots/mdot_phib_t.png",bbox_inches='tight')
        plt.close(fig)
    else:
        return ax

def extract_outflow_parameters(pkl):
    # extract outflow parameters such as efficiency, Mdot/Mdot_B, Pdot for GIZMO
    # TODO: Pdot later -> does KungYi need T^r_i for all components?
    from plotProfiles import readQuantity
    matplotlib_settings()
    
    with open(pkl, 'rb') as openFile:
        D = pickle.load(openFile)
        radii = D['radii']
        n_zones = D['nzone']
        try: n_zones_eff = D['nzone_eff']
        except: n_zones_eff = n_zones
        if n_zones_eff > 1:
            try: base = D["base"]
            except: base = 8
            try: zone_number_sequence = np.array(D["zones"])
            except: zone_number_sequence = np.array([n_zones_eff-1 - int(np.floor(np.log(radii[i][0])/np.log(base))) for i in range(len(radii))])
        else:
            zone_number_sequence = np.full(len(radii), 0)
        rB = 80.*D["r_sonic"]**2/(27.*5/3)

        profiles_Edot, invert = readQuantity(D, 'Edot')
        profiles_Mdot, invert = readQuantity(D, 'Mdot')

        r_plot = np.array([])
        Edot_save = np.array([])
        Mdot_save = np.array([])
        Mdot10_save = np.array([])
        matchingIndices = np.where(zone_number_sequence == n_zones_eff-1)[0]
        overlap = len(radii[matchingIndices[0]])//4 # overlap measured with the smallest annulus' n_radii
        for zone in range(n_zones_eff):
            matchingIndices = np.where(zone_number_sequence == zone)[0]
            radii_zone = radii[matchingIndices[0]]
            n_radii = len(radii_zone)
            mask = get_mask(zone, n_zones_eff, n_radii, overlap)
            if min(radii_zone[mask]) < rB and max(radii_zone[mask]) > rB:
                # If the radii contains Bondi radius
                #print('1e5 containing zone: '+str(zone))
                irb = np.argmin(abs(radii_zone-rB))
                for i in matchingIndices:
                    Edot_save = np.concatenate((Edot_save,np.array(profiles_Edot[i])[:,irb]))
                    Mdot_save = np.concatenate((Mdot_save,np.array(profiles_Mdot[i])[:,irb]))
            if min(radii_zone[mask]) < 10 and max(radii_zone[mask]) > 10:
                # If the radii contains r = 10
                #print('10 containing zone: '+str(zone))
                i10 = np.argmin(abs(radii_zone-10))
                for i in matchingIndices:
                    Mdot10_save = np.concatenate((Mdot10_save,np.array(profiles_Mdot[i])[:,i10]))
        
        mean_Mdot10 = np.mean(Mdot10_save[len(Mdot10_save)//2:])
        mean_Edot = np.mean(Edot_save[len(Edot_save)//2:])
        mean_Mdot = np.mean(Mdot_save[len(Mdot_save)//2:])
        mean_eta = (mean_Mdot - mean_Edot) / mean_Mdot10
        Mdot_B = bondi.get_quantity_for_rarr([1e5], 'Mdot', rs = np.sqrt(1e5))[0]

        print('mean eta = {:.5g}, Mdot/Mdot_B = {:.5g}'.format(mean_eta, mean_Mdot10/Mdot_B))


def _main():
    #dirtag="bondi_multizone_022723_jit0.3_new_coord"
    #dirtag="bondi_multizone_030723_bondi_128^3"
    #dirtag="082423_n8"
    #dirtag="2023/082423_n4"
    #dirtag="083123_ozrst_onezone"
    #dirtag="production_runs/072823_beta01_onezone"
    #dirtag="production_runs/072823_beta01_128"
    #dirtag="101123_gizmo_mhd_3d"
    #dirtag="102323_n8_constrho"
    #dirtag="2023/100223_onezone_n4"
    #dirtag="101623_n4_locff" #cap"
    #dirtag="2023/110623_n4_reproduce_082423"
    #dirtag="111123_n4_onezone_wks"
    #dirtag="111423_n4_wks"
    #dirtag="110523_mhd_wks_128"
    #dirtag="112823_n8_locff_r_gamma"
    #dirtag="122723_n4_onezone_wks0.03"
    #dirtag="011224_a0.9"
    dirtag="012224_a0.1_movingrin"
    #dirtag="012624_a0.9_movingrin_combinein_2trun"
    f=None
    #plot_snapshot(dirtag,show_mdot_phib=False)
    #mdot_phib_t('../data_products/production_runs/072823_beta01_128_profiles_all2.pkl')
    #extract_outflow_parameters('../data_products/'+dirtag+'_profiles_all2.pkl')
    FE_slice(dirtag,f)#,r_slice=[1e1,1e4])
    #FE_compare([dirtag, "082423_n8"],f,labels=[r'$128^3$', r'$64^3$'])
    #Omega_profile(dirtag,f)
    #Omega_slice(dirtag,f)
    #eta_profile(dirtag,f,boxcar_factor=4)

if __name__ == "__main__":
    _main()
