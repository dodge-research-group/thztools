import datamc
from matplotlib import pyplot as plt


def plotsignal():
    data = datamc.datamc()

    plt.figure(figsize=(10, 8))
    plt.plot(data["t"], data["y0"], color="black", label=r"Signal ($\mu$)")
    plt.plot(
        data["t"],
        data["sigma_t"] * 30,
        color="red",
        linestyle="dashed",
        label=r"noise $(30 \sigma$)",
    )
    plt.xlabel(r"Time $(ps)$", fontsize=14)
    plt.ylabel(r"Amplitude (units of $\mu_{p})$", fontsize=14)
    plt.legend(fontsize=14)
    plt.show()


plotsignal()
