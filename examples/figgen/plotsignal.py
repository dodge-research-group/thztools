from matplotlib import pyplot as plt
import datamc as datamc

# get the data from datamc
data = datamc.datamc()


def plotsignal():
    data = datamc.datamc()

    plt.figure(figsize=(10, 8))
    plt.plot(data['t'], data['y0'], color = 'black', label = 'Signal ($\mu$)')
    plt.plot(data['t'], data['sigma_t'] * 30, color = 'red' , linestyle='dashed', label = 'noise $(30 \sigma$)')
    plt.xlabel('Time $(ps)$', fontsize = 14)
    plt.ylabel('Amplitude (units of $\mu_{p})$', fontsize = 14)
    plt.legend(fontsize = 14)
    plt.show()

    return


plotsignal()