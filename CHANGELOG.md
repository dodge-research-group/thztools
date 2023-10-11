# Changelog

[![Common Changelog](https://common-changelog.org/badge.svg)](https://common-changelog.org)

Please follow the guidelines at [Common Changelog](https://common-changelog.org) for maintaining
this file.

## [0.3.1] - 2023-10-11

_Changelog introduced._

### Changed

- Use internal scaling parameters to `tdnoisefit` (Steve)

### Added

- Add changelog (Steve)
- Add `fit` function (#25) (Alireza)
- Add Notes, References, and Examples docstring sections to `noisevar`, `noiseamp`, and `thzgen` (Steve)
- Add scaling parameters to `tdnll` (now `_tdnll_scaled`) (Steve)
- Add logo (Steve)

### Removed

- Remove `costfunlsq` (Steve)
- Remove `costfuntls`, `tdnll` from public functions (rename as `_costfuntls` and `_tdnll_scaled`) (Steve)