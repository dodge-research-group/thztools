<div style="text-align: center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dodge-research-group/thztools/main/docs/source/_static/thztools_logo_dark.svg" width="600">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/dodge-research-group/thztools/main/docs/source/_static/thztools_logo.svg" width="600">
  <img alt="THzTools logo." src="https://raw.githubusercontent.com/dodge-research-group/thztools/main/docs/source/_static/thztools_logo.svg" width="600">
</picture>
</div>

THzTools is an open-source Python package for data analysis in terahertz time-domain spectroscopy
(THz-TDS). As the name suggests, THz-TDS involves measurements of terahertz-frequency
electromagnetic waveforms that are acquired as a function of *time*, which users typically
represent as a function of *frequency* for analysis. THzTools makes it easier for researchers
to use statistically optimal methods for doing this analysis, as described in [L. Mohtashemi et al.,
 Opt. Express **29**, 4912 (2021)](https://doi.org/10.1364/OE.417724).

This is *beta* software that is currently under development.

| Information       | Links                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Project**       | [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) ![GitHub](https://img.shields.io/github/license/dodge-research-group/thztools) ![PyPI - Status](https://img.shields.io/pypi/status/thztools) ![PyPI - Version](https://img.shields.io/pypi/v/thztools) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/thztools/badges/version.svg)](https://anaconda.org/conda-forge/thztools) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thztools) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) [![DOI](https://zenodo.org/badge/569133241.svg)](https://zenodo.org/doi/10.5281/zenodo.10100093) [![pyOpenSci Peer-Reviewed](https://pyopensci.org/badges/peer-reviewed.svg)](https://github.com/pyOpenSci/software-review/issues/209) [![DOI](https://joss.theoj.org/papers/10.21105/joss.07542/status.svg)](https://doi.org/10.21105/joss.07542) |
| **Build**         | ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dodge-research-group/thztools/sphinx.yml?label=build%3Adocs) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dodge-research-group/thztools/pytest-with-coverage.yml?label=build%3Atests%20(conda)) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dodge-research-group/thztools/test-pip.yml?label=build%3Atests%20(pip)) [![codecov](https://codecov.io/gh/dodge-research-group/thztools/branch/dev/graph/badge.svg?token=U8PLKTQ7AH)](https://codecov.io/gh/dodge-research-group/thztools)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Documentation** | https://dodge-research-group.github.io/thztools/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Cite**          | L. Mohtashemi et al., *Opt. Express* **29**, 4912 (2021). [(DOI)](https://doi.org/10.1364/OE.417724)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

The original MATLAB code is available at [Zenodo](https://zenodo.org/record/4876388).

## Installation

You can install THzTools with `pip`:

```shell
pip install thztools
```

THzTools is also available through `conda-forge`:

```shell
conda install -c conda-forge thztools
```

See the [Getting started](https://dodge-research-group.github.io/thztools/getting_started.html) tutorial for additional information.

## Usage

In the conventional approach to THz-TDS analysis, one transforms
the time-domain measurements into the frequency domain for further analysis.
This approach has well-known problems, however, which can be resolved by
using a maximum-likelihood estimation procedure in the time 
domain. To support this mode of analysis, the `THzTools`
package provides functionality and documentation that are unavailable in
existing THz-TDS analysis software. It provides functions to simulate THz-TDS
measurements (eg, [`timebase`](https://dodge-research-group.github.io/thztools/generated/thztools.timebase.html)
and [`wave`](https://dodge-research-group.github.io/thztools/generated/thztools.wave.html)), 
apply a frequency response function to a THz-TDS waveform ([`apply_frf`](https://dodge-research-group.github.io/thztools/generated/thztools.apply_frf.html)),
characterize the noise of a THz-TDS system ([`noisefit`](https://dodge-research-group.github.io/thztools/generated/thztools.noisefit.html)),
and fit a parameterized frequency response function to a pair of input and
output waveforms ([`fit`](https://dodge-research-group.github.io/thztools/generated/thztools.fit.html)).

With THzTools, you can:
- [Simulate a terahertz waveform](https://dodge-research-group.github.io/thztools/examples/simulate-waveform.html)
- [Simulate realistic time-domain noise for a waveform](https://dodge-research-group.github.io/thztools/examples/simulate-noise.html)
- [Apply a frequency response function (FRF) to a time-domain waveform](https://dodge-research-group.github.io/thztools/examples/apply-frequency-response.html)
- [Rescale and shift a waveform](https://dodge-research-group.github.io/thztools/examples/rescale-and-shift-waveform.html#)
- [Estimate time-domain system noise parameters from a set of measured waveforms](https://dodge-research-group.github.io/thztools/examples/estimate-noise.html)
- [Fit a parameterized FRF to a pair of time-domain waveforms](https://dodge-research-group.github.io/thztools/examples/fit-with-frf.html)

For example, the following code creates an ideal waveform and applies a
frequency response function to it.

```python
import numpy as np
import thztools as thz

# Set the waveform parameters
n = 256  # Number of samples
dt = 0.05  # Sampling time [ps]
a = 0.5 # Scale factor
eta = 1.0 # Delay [ps]

# Simulate the waveform
t = thz.timebase(n, dt=dt)
mu = thz.wave(n, dt=dt)

# Define a frequency response function
def frfun(omega, _a, _eta):
    return _a * np.exp(-1j * omega * _eta)

# Apply the frequency response function to the waveform
psi = thz.apply_frf(frfun, mu, dt=dt, args=(a, eta))
```

## Related software
Below is a list of other software projects that address related tasks in
THz-TDS analysis, with summaries taken from the project documentation.
- https://github.com/THzbiophotonics/Fit-TDS
  - "Python code aiming at the retrieving of material parameter from a
    TeraHertz time domain spectroscopy (TDS) measurements from a fit in the time
    domain."
- https://github.com/puls-lab/phoeniks
  - "A free and open-source (FOSS) Python class to extract the refractive index
    and absorption coefficient from time traces of a THz Time Domain
    Spectrometer (THz-TDS)."
- https://github.com/dotTHzTAG/CaTSper
  - "The CaTSper tool extracts the frequency-dependent optical constants from
    terahertz time-domain waveforms." (uses MATLAB)
- https://github.com/YaleTHz/nelly
  - "Nelly is a package for extracing refractice indices and conductivities from
    time-domain terahertz spectroscopy data." (uses MATLAB)

## Citation

If you use THzTools, please consider citing the **Optics Express** paper that
describes the maximum-likelihood methodology and/or the Journal of Open Source
Software (**JOSS**) paper that describes the software.

### Optics Express
```
@article{Mohtashemi2021,
author = {Laleh Mohtashemi and Paul Westlund and Derek G. Sahota and 
Graham B. Lea and Ian Bushfield and Payam Mousavi and J. Steven Dodge},
journal = {Opt. Express},
number = {4},
pages = {4912--4926},
publisher = {Optica Publishing Group},
title = {Maximum-likelihood parameter estimation in terahertz time-domain 
spectroscopy},
volume = {29},
month = {Feb},
year = {2021},
url = {https://doi.org/10.1364/OE.417724},
doi = {10.1364/OE.417724},
}
```

### JOSS
```
@article{loaiza2024,
  title = {{{THzTools}}: Data Analysis Software for Terahertz Time-Domain Spectroscopy},
  shorttitle = {{{THzTools}}},
  author = {Loaiza, Jonathan Posada and {Higuera-Quintero}, Santiago and Noori, Alireza and Mohtashemi, Laleh and Hall, R. P. and Yimam, Naod Ayalew and Dodge, J. Steven},
  year = {2024},
  journal = {Journal of Open Source Software},
  volume = {9},
  number = {104},
  pages = {7542},
  doi = {10.21105/joss.07542}
}
```
