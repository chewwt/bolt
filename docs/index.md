# Welcome to BoLT

Benchmark suite for Bayesian optimization of expensive LLM Tasks

## Installation

```bash
pip install bolt-bench
```

## Quick Start

```python
import torch
from bolt import HPO

prob = HPO(noise_std=0.001, negate=False)

X = torch.Tensor([[0, 2, 2, 2, 0.5, 30, 2]])
y = prob(X)

```

See the [Reference](api/index.md) for full documentation.
