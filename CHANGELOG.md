# Changelog

[![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org)

Please follow the guidelines at [Common Changelog](https://common-changelog.org)
for maintaining this file.

## [0.5.3] - 2024-10-12

### Fixed
- Fix documentation error in the Contributing section
  ([#59](https://github.com/dodge-research-group/thztools/issues/59))
- Fix typographical errors in JOSS paper

### Added
- Add [repostatus.org](https://www.repostatus.org/) badge
  ([#58](https://github.com/dodge-research-group/thztools/issues/58))
- Add documentation on type checking in the Contributing section
  ([#59](https://github.com/dodge-research-group/thztools/issues/59))
- Add student acknowledgements to JOSS paper

## [0.5.2] - 2024-09-11

### Changed
- Replace Black badge with Ruff badge in `README.md`

### Fixed
- Fix documentation error in `fit`

## [0.5.1] - 2024-08-15

### Added
- Add `CONTRIBUTING.md` file
- Add *Installation* section to `README.md`

### Fixed
- Fix error in `bug_report` issue template

## [0.5.0] - 2024-08-01

### Changed
- Change development status to beta
- Change `mu_var` attribute of `FitResult` to `mu_err`
- Change `p_var` attribute of `FitResult` to `p_cov`
- Revise example of simulating noise
- Revise example of simulating a waveform
- Complete draft JOSS paper
- Rename `transfer` as `apply_frf`
- Revise docstrings for consistency and clarity
- Apply bias correction to noise model parameter estimates
- Reorganize tests

### Added
- Add configuration file for doctests with PyTest
- Add example of appling a frequency response function
- Add example of estimating noise model parameters
- Add example of fitting a parameterized frequency response function model
- Add example of rescaling and shifting a waveform
- Add `diagnostic`, `dof`, `frfun_opt`, `p_err`, and `psi_opt` attributes to
  `FitResult`
- Add frequency bounds to `fit`
- Add tests for new functionality

### Removed
- Remove `examples` directory from top level
- Remove unused lint options from `pyproject.toml`

### Fixed
- Correct total least-squares residual computation

## [0.4.2] - 2024-06-30

### Changed

- Revise docstrings for `NoiseModel` methods
- Refactor `noisefit` code to improve readability
- Set tolerance parameter in `minimize` function used in `noisefit`
- Change `fft_sign` options to `numpy_sign_convention`
- Restore factor of `0.5` in cost function computation and gradient
- Replace `scale_sigma_*` parameters (which simply adds a constant to the log) 
  with `scale_logv_*` parameters
- Refactor `noisefit` to encapsulate cost function computations in separate
  functions, `_nll_noisefit`, `_jac_noisefit`, and `_hess_noisefit`

### Added

- Add `reset_option`, `set_option`
- Rename `simulate-ideal-waveform` example to `simulate-waveform`
- Add `conftest.py` file to ensure that global options are reset with each test
- Add `doctest` to `test-pip` action
- Add `doctest` script and dependencies to `pyproject.toml`
- Add test to verify noise parameter estimates in `TestNoiseFit`
- Add `_nll_noisefit`, `_jac_noisefit`, and `_hess_noisefit`

### Fixed

- Fix error in `wave` docstring example
- Fix scaling error in `noisefit` estimate for `sigma_tau`
- Fix `noisefit` docstring example
- Fix error in `fit` covariance estimate output

## [0.4.1] - 2024-05-25

### Changed

- Refactor `options`
- Use explicit `dt` keyword option instead of global option in examples
- Change hatch doctest script from Sphinx to PyTest
- Change signature of `fit`
- Revise documentation for `fit`, `FitResult`, `get_option`, `set_option`,
  `transfer`
- Refactor `_costfuntls`

### Added

- Add version badges for pip and Anaconda
- Add conda installation instructions
- Add `FitResult` dataclass and use it to return `fit` output
- Add `autouse` fixture to reset `sampling_time` option to default before each
  test
- Add Python 3.12 to testing workflow
- Add global doctest imports explicitly to docstrings
- Add `numpy_sign_convention` parameter to `fit`

### Fixed

- Fix scaling error in `_parse_noisefit_output`

## [0.4.0] - 2024-05-10

### Changed

- Rename module constant `global_options` as `options`
- Rename `tdnoisefit` as `noisefit`
- Rename `NoiseModel` attributes `alpha`, `beta` and `tau` as `sigma_alpha`,
  `sigma_beta` and `sigma_tau`, respectively
- Rename `NoiseModel` methods `amplitude`, `noise`, and `variance` as
  `noise_amp`, `noise_sim`, and `noise_var`, respectively
- Revise `noisefit` and associated functions to treat each of the 3 noise
  parameters separately instead of as a 3-component array, so that each may
  fixed independently in the optimization process
- Adjust internal scaling in `noisefit`
- Reorganize `NoiseResult` class as output for `noisefit`
- Revise `NoiseResult` docstring
- Rename `transfer_out` as `transfer`
- Rename `fft_sign` parameter as `numpy_sign_convention` in `transfer`
- Make `dt` and `t0` keyword-only arguments in `wave` and set defaults
- Change position of `dt` in `scaleshift`, `tdnoisefit`, `transfer` docstrings
  and change defaults
- Revise defaults for `wave`
- Change default value for `dt` in `fit`
- Improve type annotations
- Update tests
- Update dependencies in `environment-dev.yml` and `pyproject.toml`
- Revise *Getting Started* page
- Revise documentation, including Sphinx format and project-specific layout

### Added

- Add R. P. Hall, Laleh Mohtashemi and Naod Yimam to the author list
- Add missing ORCID IDs
- Add `get_option` and `set_option` functions to handle global options
- Add attributes to `NoiseResult` class
- Add `timebase` function
- Add `seed` parameter in `NoiseModel.noise_sim`
- Add `NoiseResult` class to documentation
- Add doctests
- Add warnings
- Add `paper` directory with bib-file and stub for JOSS paper
- Add `draft-pdf.yml` GitHub Action to autogenerate JOSS paper; update
  deprecated `upload-artifact` Action in workflow
- Add Contributing and Examples pages to documentation
- Add Code of Conduct
- Add gallery of Jupyter notebook examples to documentation using `nbsphinx`

### Fixed

- Fix error in units of `NoiseModel.sigma_tau` in `noisefit`
- Fix bug related to `eta` scaling in `noisefit`

## [0.3.6] - 2023-11-12

### Changed

- Reorganized `NoiseResult` dataclass
- Initialize tests with `thztools.global_options.sampling_time = None`

### Added

- Add [Zenodo](https://zenodo.org/records/10115206) badge to README
- Internal `_validate_sampling_time` function to handle sampling time input
- Tests for `_validate_sampling_time`

## [0.3.5] - 2023-11-11

### Added

- Add `GlobalOptions` class
- Add `global_options` module constant

## [0.3.4] - 2023-11-10

### Changed

- Fix bug in `NoiseModel` example
- Replace tuple return value in `tdnoisefit` with `NoiseResult`

### Added

- `NoiseResult` class

## [0.3.3] - 2023-11-09

### Changed

- Prune and update notebooks
- Fix error in `noiseamp` docstring example (now `amplitude` method of
  `NoiseModel` class)
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
- Add Notes, References, and Examples docstring sections to `noisevar`,
  `noiseamp`, and `thzgen` (Steve)
- Add scaling parameters to `tdnll` (now `_tdnll_scaled`) (Steve)
- Add logo (Steve)

### Removed

- Remove `costfunlsq`
- Remove `costfuntls`, `tdnll` from public functions (rename as `_costfuntls`
  and `_tdnll_scaled`)

## [0.3.0] - 2023-09-07

### Added

- Add `fit` function (#25) (Alireza)
- Add pytest classes for all functions (Steve)

### Removed

- Remove `tdtf`

[0.5.3]: https://github.com/dodge-research-group/thztools/releases/tag/v0.5.3
[0.5.2]: https://github.com/dodge-research-group/thztools/releases/tag/v0.5.2
[0.5.1]: https://github.com/dodge-research-group/thztools/releases/tag/v0.5.1
[0.5.0]: https://github.com/dodge-research-group/thztools/releases/tag/v0.5.0
[0.4.2]: https://github.com/dodge-research-group/thztools/releases/tag/v0.4.2
[0.4.1]: https://github.com/dodge-research-group/thztools/releases/tag/v0.4.1
[0.4.0]: https://github.com/dodge-research-group/thztools/releases/tag/v0.4.0
[0.3.6]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.6
[0.3.5]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.5
[0.3.4]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.4
[0.3.3]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.3
[0.3.2]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.2
[0.3.1]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.1
[0.3.0]: https://github.com/dodge-research-group/thztools/releases/tag/v0.3.0