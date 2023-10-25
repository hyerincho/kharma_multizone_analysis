import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt

def matplotlib_settings():
    """
    Makes some modifications to the default matplotlib settings.
    """
    #rc('font',**{'family':'serif','serif':['Times']})
    #rc('text', usetex=True)
    #rc('font', size=40)
    #rc('axes', titlesize=40)
    #rc('legend', fontsize=40)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams.update({'font.size': 18})#, "text.usetex": True}) # 18
    #plt.rcParams["font.weight"] = "bold"
    #plt.rcParams["axes.labelweight"] = "bold"
