# BoLT

A benchmark suite for Bayesian optimization of expensive LLM tasks. Each problem is backed by a pretrained neural-network surrogate or tabular data from real LLM experiments, so evaluations are fast and reproducible without running real LLM training.

## Installation

```bash
pip install bolt-bench
# or with uv
uv pip install bolt-bench
```

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
prob = HPO(noise_std=0.001, negate=False)

X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2]])  # one candidate configuration
y = prob(X)  # shape: (1,)
```

## Problems

| Family | Classes | Notes |
|---|---|---|
| [Hyperparameter optimization](api/hpo.md) | `HPO`, `HPOMultiFidelityToken`, `HPOMultiFidelityModel` | mixed continuous/discrete/categorical; multi-fidelity variants available |
| [Data mixture](api/dm.md) | `DMCurriculum`, `DMCurriculumMO`, `DMCurriculumHet` | simplex-constrained inputs; multi-objective and heteroscedastic variants |
| [Prompt optimization](api/po.md) | `PO128`, `PO256`, `PO512`, `PO768` | high-dimensional continuous embedding search (128–768 dims) |
| [Parallelism configuration](api/pco.md) | `PCO16`, `PCO32`, `PCO64` | black-box constraint over a discrete candidate set |

See [Problems](problems.md) for full details on inputs, constraints, and fidelity parameters.

## Next Steps

- [Problems](problems.md) — detailed problem descriptions and parameter tables
- [API Reference](api/index.md) — full class and method documentation
