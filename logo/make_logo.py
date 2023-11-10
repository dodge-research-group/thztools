import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from thztools import wave

n = 1024
ts = 0.01
t0 = 1
mu, t = wave(n, ts, t0)

# Use the Multicolored Lines example in the Matplotlib documentation to produce
# the rainbow-colored waveform.
# https://matplotlib.org/stable/gallery/lines_bars_and_markers
# /multicolored_line.html
points = np.array([t, mu]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

cmap = "nipy_spectral"
fontname = "Arial"
fontweight = "bold"
fontstyle = "normal"
figsize = [5.6, 1.4]

_, ax = plt.subplots(figsize=figsize, layout="constrained")

norm = plt.Normalize(t.min(), t.max())
lc = LineCollection(segments, cmap=cmap, norm=norm, clip_on=False)

lc.set_array(t)
lc.set_linewidth(4)
line = ax.add_collection(lc)

ax.text(
    2.0,
    0,
    "THzTools",
    fontname=fontname,
    fontweight=fontweight,
    fontstyle=fontstyle,
    verticalalignment="bottom",
    fontsize=48,
    color="#202328",
)
ax.axis("off")
ax.set_ylim(mu.min(), mu.max())
ax.set_xlim(t.min(), t.max())
plt.savefig("thztools_logo.svg", transparent=True)

del ax, norm, lc, line

_, ax = plt.subplots(figsize=figsize, layout="constrained")

norm = plt.Normalize(t.min(), t.max())
lc = LineCollection(segments, cmap=cmap + "_r", norm=norm, clip_on=False)

lc.set_array(t)
lc.set_linewidth(4)
line = ax.add_collection(lc)

ax.text(
    2.0,
    0,
    "THzTools",
    fontname=fontname,
    fontweight=fontweight,
    fontstyle=fontstyle,
    verticalalignment="bottom",
    fontsize=48,
    color="#E7EDF2",
)
ax.axis("off")
ax.set_ylim(mu.min(), mu.max())
ax.set_xlim(t.min(), t.max())
plt.savefig("thztools_logo_dark.svg", transparent=True)
