# Changelog

[![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org)

Please follow the guidelines at [Common Changelog](https://common-changelog.org) for maintaining
this file.

## [0.3.3] - 2023-11-09

### Changed

- Prune and update notebooks
- Fix error in `noiseamp` docstring example (now `amplitude` method of `NoiseModel` class)
- Darken text color for logo in dark mode
- Move make-logo script into default hatch environment
- Rename sampling time variable from `ts` to `dt`
- Rename `thzgen` as `wave`
- Revise `conf.py` in Sphinx documentation
- Add SciPy templates for `attributes.rst`, `class.rst` and `method.rst`
- Refactor code to decouple computations from sampling time
- Make `dt` parameter optional in most functions
- Set default `dt` to `None`, to indicate that time units are unspecified

### Added

- `NoiseModel` class
- `transfer_out` function
- Example for `scaleshift` function
- Specify logo based on light/dark theme

### Removed

- `noiseamp` and `noisevar` functions

## [0.3.2] - 2023-10-11

### Changed

- Point logo link to raw image file on GitHub

## [0.3.1] - 2023-10-11

_Changelog introduced._

### Changed

- Use internal scaling parameters to `tdnoisefit` (Steve)

### Added

- Add changelog (Steve)
- Add frequency bounds to fit function (#27) (Alireza)
- Add Notes, References, and Examples docstring sections to `noisevar`, `noiseamp`, and `thzgen` (Steve)
- Add scaling parameters to `tdnll` (now `_tdnll_scaled`) (Steve)
- Add logo (Steve)

### Removed

- Remove `costfunlsq`
- Remove `costfuntls`, `tdnll` from public functions (rename as `_costfuntls` and `_tdnll_scaled`)

## [0.3.0] - 2023-09-07

### Added

- Add `fit` function (#25) (Alireza)
- Add pytest classes for all functions (Steve)

### Removed

- Remove `tdtf`

[0.3.3]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.3
[0.3.2]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.2
[0.3.1]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.1
[0.3.0]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.0