import matplotlib.pyplot as plt

def matplotlib_settings():
    """
    Makes some modifications to the default matplotlib settings.
    """

    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
