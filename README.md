<img src="https://raw.githubusercontent.com/chewwt/bolt/main/bolt_logo.png" width="300"/>

[![CI](https://github.com/chewwt/bolt/actions/workflows/ci.yml/badge.svg)](https://github.com/chewwt/bolt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/chewwt/bolt/branch/main/graph/badge.svg)](https://codecov.io/gh/chewwt/bolt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

A benchmark suite for Bayesian optimization of expensive LLM tasks. Each problem is backed by a pretrained neural-network surrogate or tabular data from real LLM experiments, so evaluations are fast and reproducible without running real LLM training.


## Documentation

Full documentation is available at [bolt-bench.readthedocs.io](https://bolt-bench.readthedocs.io/en/latest/).

## Installation

`pip install bolt-bench`

Use **0.1.1 or later**. From 0.1.1, each release pins a specific emulator revision. Earlier versions load emulator weights from the HuggingFace `main` branch, so their results shift whenever new weights are published.

| bolt | emulator revision |
|---|---|
| ≤ 0.1.0 | tracks `main` (unpinned) |
| 0.1.1 | `v0.1.0` |
| 0.2.0 | `v0.2.0` (HPO and data mixture), `v0.1.0` (prompt optimization and parallelism configuration) |

## Quick Start

```python
import torch
from bolt import HPO

# 7-dim HPO problem: returns a scalar surrogate of eval score
prob = HPO()

X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2]])  # one candidate configuration
y = prob(X)  # shape: (1, 1)
```


## Problems


| Problem | Class | Dims | Notes |
|---|---|---|---|
| HPO | `HPO` | 7 | mixed params (continuous, discrete, categorical) |
| HPO multi-fidelity (token) | `HPOMultiFidelityToken` | 8 | mixed params (continuous, discrete, categorical), fidelity: continuous ∈ [0, 1] (training tokens) |
| HPO multi-fidelity (model) | `HPOMultiFidelityModel` | 8 | mixed params (continuous, discrete, categorical), fidelity: discrete ∈ {0, 1} (model size) |
| Data mixture | `DMCurriculum` | 6 | two simplex constraints |
| Data mixture MO | `DMCurriculumMO` | 6 | two simplex constraints, multi-objective (3) |
| Data mixture with heteroscedastic noise | `DMCurriculumHet` | 6 | two simplex constraints, input-dependent noise |
| Prompt optimization (128-dim) | `PO128` | 128 | discrete candidate set |
| Prompt optimization (256-dim) | `PO256` | 256 | discrete candidate set |
| Prompt optimization (512-dim) | `PO512` | 512 | discrete candidate set |
| Prompt optimization (768-dim) | `PO768` | 768 | discrete candidate set |
| Parallelism configuration (16 GPUs) | `PCO16` | 8 | discrete candidate set, black-box constraint |
| Parallelism configuration (32 GPUs) | `PCO32` | 8 | discrete candidate set, black-box constraint |
| Parallelism configuration (64 GPUs) | `PCO64` | 8 | discrete candidate set, black-box constraint |

### Observation noise

The HPO and data mixture problems are **noisy by default**, using an empirically measured noise level: `noise_std` defaults to `_measured_std`, the standard deviation observed across repeat training runs of the real task. This matches the setting the benchmark results were produced under, so `HPO()` reproduces it without extra arguments.

Set `noise_std` to override it with a value of your own for a different noise level or `None` for a noiseless problem.

```python
prob = HPO()                    # noise_std = 0.0290, the measured value
prob = HPO(noise_std=None)      # noiseless
prob = HPO(noise_std=0.01)      # your own level

prob(X)                         # noisy
prob(X, noise=False)            # noiseless, whatever the default
```

`DMCurriculumHet` is the exception: its MATH-500 noise is input-dependent and comes from a noise emulator, so `noise_std` is ignored and `prob.evaluate_noise(X)` returns the std at `X`. The IFEval and MBPP+ stds are held at measured constants; see [Observation noise](https://bolt-bench.readthedocs.io/en/latest/problems/#observation-noise).
