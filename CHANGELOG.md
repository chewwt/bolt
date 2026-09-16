## [0.1.1] - 2026-09-16
### Added
- `revision` argument on `pull_info_from_hf_hub` and the tabular loaders
- `hf_revision` class attribute on problems, pinning their emulator weights

### Changed
- Emulator weights load from the pinned revision instead of tracking `main`

### Fixed
- "NumPy array is not writable" warning from pandas copy-on-write

## [0.1.0] - 2026-05-03
### Added
- Initial release
