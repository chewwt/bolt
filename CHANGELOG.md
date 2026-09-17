## [Unreleased]

### Added
- `n_layers` argument on `FeatureNet` and `MLPFunction`, read from the emulator
  `config.json`. Required by the v0.2.0 emulators, where layers are specified by the config
- `gamma` argument on `KRFunction`, read from the noise emulator `config.json`
- `pull_noise_info_from_hf_hub`, which fetches the noise emulator config
  alongside its weights
- `_measured_std` on each problem: the observation noise empirically measured
  across repeat training runs of the real task

### Changed
- `HPO`, `HPOMultiFidelityToken`, `HPOMultiFidelityModel`, `DMCurriculum` and
  `DMCurriculumMO` are noisy by default: `noise_std` now defaults to the
  empirically measured `_measured_std` rather than `None`, which is the setting the
  benchmark results were produced under. Pass `noise_std=None` for the previous
  default noiseless behaviour. `DMCurriculumHet` is unaffected — it ignores
  `noise_std` and uses its noise emulator
- HPO and data mixture emulators pinned to `v0.2.0` (prompt optimization stays on
  `v0.1.0`). The new emulators are deeper and wider, and more accurate
- `_optimal_value` and `_optimizers` re-measured against the v0.2.0 weights for
  `HPO`, `HPOMultiFidelityToken`, `HPOMultiFidelityModel`, `DMCurriculum` and
  `DMCurriculumHet`; `_ref_point` and `_max_hv` re-measured for `DMCurriculumMO`
- `DMCurriculumHet` objective is now the mean of IFEval, MATH-500 and MBPP+
  rather than MATH-500 alone. Its noise combines the MATH-500 noise emulator with
  the measured IFEval and MBPP+ stds
- `DMCurriculumHet` noise emulator now takes all 6 curriculum parameters; v0.1.0
  was fit on 3 of them
- `DMCurriculumHet` noise is no longer scaled by 0.1. The emulator predicts the
  std directly, so the previously reported noise was 10x too small
- `HPOMultiFidelityToken.cost` is now `0.95 * fidelity + 0.05`, derived from
  training FLOPs being linear in tokens seen, with a 5% fixed per-run overhead
- `HPOMultiFidelityModel.cost` is now `0.47 * fidelity + 0.53`, derived from the
  FLOPs-per-token ratio of Qwen3-4B to Qwen3-8B. The 4B fidelity is ~2x cheaper,
  not 10x

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
