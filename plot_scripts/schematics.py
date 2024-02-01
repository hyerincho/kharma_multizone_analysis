import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_settings import *
import pdb

def schematic(n_zones=8, base=8, fake=True, rs=np.sqrt(1e5), gam=5./3):
    matplotlib_settings()
    #fig, ax = plt.subplots(1, 1, figsize=(12,6))
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    log_r_in = np.log10(np.array([base**i for i in range(n_zones)])) # array of log_r_ins
    t_run = 0
    zone_order = [n_zones-1-i for i in range(n_zones)] + [i+1 for i in range(n_zones-1)] # one V cycle
    colors=plt.cm.gnuplot(np.linspace(0.3,0.9,n_zones))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    y = [(i+1) for i in range(n_zones)]
    yticks = [r'$10^{:d}$'.format(i) for i in y]
    r_B = 80. * rs**2 / (27 * gam)
    
    for nz in range(len(zone_order)):
        zone = zone_order[nz]
        if fake: tchar = np.power(1.5, zone) # this is the fake part
        else: 
            r_out = np.power(10.,log_r_in[zone]) * base**2
            vff2 = 1. / r_out
            vcs2 = 1./ r_B
            tchar = r_out / np.sqrt(vff2 + vcs2) #np.power(np.power(10,log_r_in[zone]),3./2)
        edgecolor = 'k'; lw=None; zorder=None
        if nz == len(zone_order) - 2 or nz == 1 or nz == len(zone_order) - 3: 
            log_r_in_temp = log_r_in[n_zones - 2]
            if nz == 1: t_save = t_run + tchar
            if nz == len(zone_order) - 2: 
                edgecolor = 'b'; lw=5; zorder=100
                #ax.plot([t_save, t_run], [log_r_in[zone] + 1.5 * np.log10(base)]*2, ls=':', color='b')
                ax.fill_between([t_save, t_run], log_r_in_temp + np.log10(base), log_r_in_temp + np.log10(base**2), facecolor='b', alpha=0.2)#"none", edgecolor='b', ls=':')
                ax.arrow((t_save + t_run) * 0.45, log_r_in[zone] + 1.5 * np.log10(base), 8, 0, width=0.2, head_width=0.4, head_length=1, color='b', alpha=0.2)
            if nz != 1: ax.fill_between([t_run, t_run + tchar], log_r_in_temp, log_r_in_temp + np.log10(base), facecolor="none", edgecolor='b', hatch='.', zorder=200, lw=0)
            if nz != len(zone_order) - 3: ax.fill_between([t_run, t_run + tchar], log_r_in_temp + np.log10(base), log_r_in_temp + np.log10(base**2), facecolor="none", edgecolor='b', hatch='/', zorder=200, lw=0)
        rect = matplotlib.patches.Rectangle((t_run, log_r_in[zone]), tchar, np.log10(base**2), facecolor=colors[zone], edgecolor=edgecolor, lw=lw, zorder=zorder)
        t_run += tchar
        if zone == 0 or zone == n_zones - 1:
            ax.text(t_run - tchar / 2. - 4, 8.5, "zone-{}".format(zone))
            ax.plot([t_run - tchar / 2.] * 2, [log_r_in[zone] + np.log10(base**2), 8.3], ls=':', color="gray")

        ax.add_patch(rect)

    xmin = 0; xmax = t_run + 5
    ymin = 0; ymax = np.log10(base**(n_zones+1))
    ax.set_yticks(y,yticks)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, 8.5])
    ax.set_xticks([])
    fs=23
    ax.set_xlabel('t', fontsize=fs, labelpad=15)
    ax.set_ylabel(r'$r$ [$r_g$]', fontsize=fs)
    ax.text(t_run + 1, 4,'...', fontsize=30)
    ax.arrow(t_run / 4., 8.5, 5, 0, width=0.05, head_width=0.2, head_length=1, zorder=100, clip_on=False, color='k')
    ax.arrow(t_run / 3. * 2., 8.5, 5, 0, width=0.05, head_width=0.2, head_length=1, zorder=100, clip_on=False, color='k')
    ax.axhline(np.log10(r_B), color='grey', lw=5, alpha=0.5) #, ls='--')
    ax.text(1, np.log10(r_B) - 0.5,r'$R_B$', fontsize=20, color='grey')
    
    # manual arrowhead width and length
    hw = 1./40.*(ymax-ymin) 
    hl = 1./40.*(xmax-xmin)
    lw = 0.5 # axis line width
    ohg = 0.3 # arrow overhang
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
            head_width=hw, head_length=hl, overhang = ohg, 
                length_includes_head= True, clip_on = False)

    plt.savefig('../plots/schematic.png', bbox_inches='tight')

def _main():
    schematic()#fake=False)

if __name__ == "__main__":
    _main()

