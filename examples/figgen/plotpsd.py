import datamc
import matplotlib.pyplot as plt
import numpy as np


def plotpsd(data=None):
    data = datamc.datamc()

    # Parse inputs
    nf = data["Nf"]
    f = data["f"][0:nf]
    ym = data["Ym"][0:nf, :]

    # plot
    plt.figure(figsize=(10, 8))
    plt.plot(
        f,
        10 * np.log10(abs(ym[:, 1:]) ** 2 / max(abs(ym.flatten() ** 2))),
        color="grey",
    )
    plt.plot(
        f,
        10 * np.log10(abs(ym[:, 0]) ** 2 / max(abs(ym.flatten() ** 2))),
        color="red",
    )
    plt.xlabel("Frequency (THz)", fontsize=14)
    plt.ylabel("Relative Power (dB)", fontsize=14)
    # plt.ylim([-70, 10])
    plt.show()


plotpsd()
