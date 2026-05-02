![](bolt_logo.png)

A benchmark suite for Bayesian optimization of expensive LLM Tasks


## Installation

`pip install -e .`

## Quick Start

```python
import torch
from bolt import HPO

prob = HPO(noise_std=0.001, negate=False)

X = torch.rand((1, prob.dim))
bounds = torch.Tensor(prob._bounds)
X = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

y = prob(X)

```
<!-- 
## Documentation

Full docs at https://your-username.github.io/your-library -->


## Problems


| Problem | Class | Dims | Notes |
|---|---|---|---|
| HPO | `HPO` | 7 | mixed params (discrete, categorical) |
| HPO multi-fidelity (token) | `HPOMultiFidelityToken` | 8 | mixed params (continuous, discrete, categorical), fidelity: continuous ∈ [0, 1] (training tokens) |
| HPO multi-fidelity (model) | `HPOMultiFidelityModel` | 8 | mixed params (continuous, discrete, categorical), fidelity: discrete ∈ {0, 1} (model size) |
| Data mixture | `DMCurriculum` | 6 | simplex constraint |
| Data mixture MO | `DMCurriculumMO` | 6 | simplex constraint, multi-objective (3) |
| Prompt optimization (128-dim) | `PO128` | 128 | discrete candidate set |
| Prompt optimization (256-dim) | `PO256` | 256 | discrete candidate set |
| Prompt optimization (512-dim) | `PO512` | 512 | discrete candidate set |
| Prompt optimization (768-dim) | `PO768` | 768 | discrete candidate set |
