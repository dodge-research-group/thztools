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

| Information       | Links                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|:------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Project**       | [![DOI](https://zenodo.org/badge/569133241.svg)](https://zenodo.org/doi/10.5281/zenodo.10100093) ![PyPI - Status](https://img.shields.io/pypi/status/thztools) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/thztools) ![PyPI - Version](https://img.shields.io/pypi/v/thztools) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/thztools/badges/version.svg)](https://anaconda.org/conda-forge/thztools) ![GitHub](https://img.shields.io/github/license/dodge-research-group/thztools) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) |
| **Build**         | ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dodge-research-group/thztools/sphinx.yml?label=build%3Adocs) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dodge-research-group/thztools/pytest-with-coverage.yml?label=build%3Atests%20(conda)) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dodge-research-group/thztools/test-pip.yml?label=build%3Atests%20(pip)) [![codecov](https://codecov.io/gh/dodge-research-group/thztools/branch/dev/graph/badge.svg?token=U8PLKTQ7AH)](https://codecov.io/gh/dodge-research-group/thztools)                                                                                                                                                                                                        |
| **Documentation** | https://dodge-research-group.github.io/thztools/                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Cite**          | L. Mohtashemi et al., *Opt. Express* **29**, 4912 (2021). [(DOI)](https://doi.org/10.1364/OE.417724)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

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