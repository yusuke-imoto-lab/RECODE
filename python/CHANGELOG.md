# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- Placeholder for upcoming features.

---

## [2.1.0] - 2025-06-24
### Added
- Added highly variable gene selection function `highly_variable_genes`.
- Added tutorial [Transcriptome (scRNA-seq data) + HVG selection](https://yusuke-imoto-lab.github.io/RECODE/Tutorials/Tutorial_RNA_HVG.html) on Sphinx.

---

## [2.0.0] - 2025-06-23
### Changed
- Changed the default version of noise reduction is 2.

### Fixed
- Fixed some errors in Readme.  

---

## [1.0.1] - 2025-04-08
### Added
- Added preprint information to the README.
- Added the number of invalid cells in `report()`.

### Changed
- Ignored invalid cells whose components are all zeros.

### Fixed
- Fixed bug in `check_applicability()` where class names were not correctly displayed.
- Fixed bug in `report()` where the number of cells were not correctly displayed.