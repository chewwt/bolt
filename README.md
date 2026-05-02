![](bolt_logo.png)

[![CI](https://github.com/chewwt/bolt/actions/workflows/ci.yml/badge.svg)](https://github.com/chewwt/bolt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/chewwt/bolt/branch/main/graph/badge.svg)](https://codecov.io/gh/chewwt/bolt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)

A benchmark suite for Bayesian optimization of expensive LLM tasks. Each problem is backed by a pretrained neural-network surrogate or tabular data from real LLM experiments, so evaluations are fast and reproducible without running real LLM training.


## Installation

`pip install bolt-bench`

## Quick Start

```python
import torch
from bolt import HPO

# 7-dim HPO problem: returns a scalar surrogate of eval score
prob = HPO(noise_std=0.001, negate=False)

X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2]])  # one candidate configuration
y = prob(X)  # shape: (1,)
```


## Problems


| Problem | Class | Dims | Notes |
|---|---|---|---|
| HPO | `HPO` | 7 | mixed params (continuous, discrete, categorical) |
| HPO multi-fidelity (token) | `HPOMultiFidelityToken` | 8 | mixed params (continuous, discrete, categorical), fidelity: continuous ∈ [0, 1] (training tokens) |
| HPO multi-fidelity (model) | `HPOMultiFidelityModel` | 8 | mixed params (continuous, discrete, categorical), fidelity: discrete ∈ {0, 1} (model size) |
| Data mixture | `DMCurriculum` | 6 | two simplex constraints |
| Data mixture MO | `DMCurriculumMO` | 6 | two simplex constraints, multi-objective (3) |
| Data mixture with heteroscedastic noise | `DMCurriculumHet` | 6 | two simplex constraints, heteroscedastic noise |
| Prompt optimization (128-dim) | `PO128` | 128 | discrete candidate set |
| Prompt optimization (256-dim) | `PO256` | 256 | discrete candidate set |
| Prompt optimization (512-dim) | `PO512` | 512 | discrete candidate set |
| Prompt optimization (768-dim) | `PO768` | 768 | discrete candidate set |
