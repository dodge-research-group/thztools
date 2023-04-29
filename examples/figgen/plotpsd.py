import numpy as np
import matplotlib.pyplot as plt
import datamc as datamc

def plotpsd(Data=None, ax=None):

    data = datamc.datamc()

    # Parse inputs
    Nf = data['Nf']
    f = data['f'][0:Nf]
    Ym = data['Ym'][0:Nf, :]

    # plot
    plt.figure(figsize=(10, 8))
    plt.plot(f, 10 * np.log10(abs(Ym[:, 1:]) ** 2 / max(abs(Ym.flatten() ** 2))), color = 'grey', )
    plt.plot(f, 10 * np.log10(abs(Ym[:, 0]) ** 2 / max(abs(Ym.flatten() ** 2))), color = 'red')
    plt.xlabel('Frequency (THz)', fontsize = 14)
    plt.ylabel('Relative Power (dB)', fontsize = 14)
    #plt.ylim([-70, 10])
    plt.show()

    return

plotpsd()