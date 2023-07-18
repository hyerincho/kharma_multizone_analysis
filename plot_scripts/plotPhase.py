import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pdb

from plotProfiles import readQuantity
      

def phasePlot(D, quantity, roundtrip=0, tag=None):
    profiles, invert = readQuantity(D, quantity)
    radii = D['radii']
    times = D['times']
    
    fig, ax = plt.subplots(1, 1, figsize=(8,6))

    #ax.plot(radii[0],profiles[0][0])
    #ax.set_xscale('log'); ax.set_yscale('symlog')

    norm=colors.SymLogNorm(linthresh=1e-4,vmin=-1, vmax=1)
    #norm=colors.SymLogNorm(linthresh=1e-6,vmin=-1e-2, vmax=1e-2)
    n_zones = D['nzone']
    try: r_sonic = D["r_sonic"]
    except: r_sonic = np.sqrt(1e5)
    
    num_per_round = 2*(n_zones-1)
    start = roundtrip*num_per_round
    num_per_round *= 50 #40 # plot more
    listOfIndices = np.arange(start, start+num_per_round)

    y = None
    if quantity == 'Mdot' or quantity=='Edot':
        cmap = 'RdBu'
    else: cmap = 'RdBu_r'

    for i in listOfIndices:
        if 1: #radii[i][0]<r_sonic**2: # only focus inside the Bondi radius
            x = np.log10(radii[i])
            #y = np.array(times[i])
            if y is None:
                y=[0] #times[i][0]]
            y = np.arange(y[-1],y[-1]+len(times[i]))
            X, Y = np.meshgrid(x, y)
            
            pcm = ax.pcolormesh(X,Y, np.array(profiles[i]), norm=norm, cmap=cmap)
    cb = fig.colorbar(pcm, ax=ax, extend='both')
    cb.set_label(quantity)
    ax.axvline(np.log10(2),color='k')
    ax.axvline(np.log10(r_sonic**2),color='k',ls=':')
    #ax.set_xlim(right=5)
    ax.set_xlabel(r'$\log_{\rm 10}(r)$'); ax.set_ylabel('# of outputs')
    ax.set_title('t = {:.3g} - {:.3g}'.format(times[listOfIndices[0]][0], times[listOfIndices[-1]][-1]))
    if tag is not None:
        fig.suptitle(tag)
    
    plt.savefig('./phase_plot_{:05d}.png'.format(roundtrip),bbox_inches='tight')


if __name__ == '__main__':
    #tag="062023_0.02tff_ur0"
    #tag="062223_difftchar"
    #tag="070823_b20n4_32"
    #tag="070923_b6n7_32"
    tag="071023_beta01"
    #tag="071323_beta01_64"
    #pkl = '../data_products/060223_bflux0_n5_64^3_profiles_all2.pkl'
    #pkl = '../data_products/062023_0.02tff_ur0_profiles_all.pkl'
    #pkl = '../data_products/062723_b3_0.04tff_profiles_all2.pkl'
    #pkl = '../data_products/062623_n3_tff_profiles_all2.pkl'
    #pkl = '../data_products/071023_b3n7_profiles_all2.pkl'
    pkl = '../data_products/'+tag+'_profiles_all2.pkl'
    with open(pkl, 'rb') as openFile:
      D = pickle.load(openFile)

    for i in range(100):
        print(i)
        phasePlot(D, 'Mdot',50*i,tag)
    
