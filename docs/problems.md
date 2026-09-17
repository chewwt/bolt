# Problems

A catalogue of LLM optimization problems are available in this library. 

Problems are backed by one or more pretrained emulators or tabular data downloaded automatically from HuggingFace Hub and cached locally for subsequent use. Emulator accuracy varies by problem — for Spearman's rank correlation coefficients and validation methodology see our [paper](https://arxiv.org/abs/2605.17000).
 
For full parameter details see the [API Reference](./api/index.md).
 
---

## Problem Types

| Type | Description | Example Problems |
|------|-------------|-----------------|
| Mixed-variable| Search spaces with both continuous and discrete/categorical variables | [HPO](./api/hpo.md#bolt.HPO) |
| Mixed-variable + multi-fidelity | Combines mixed variables with multiple accuracy/cost levels | [HPOMultiFidelityToken](./api/hpo.md#bolt.HPOMultiFidelityToken), [HPOMultiFidelityModel](./api/hpo.md#bolt.HPOMultiFidelityModel) |
| Simplex constrained | Search space is constrained by two simplices | [DMCurriculum](./api/dm.md#bolt.DMCurriculum)|
| Simplex constrained + multi-objective | Two or more objectives to optimize simultaneously | [DMCurriculumMO](./api/dm.md#bolt.DMCurriculumMO) |
| Simplex constrained + heteroscedastic noise | Noise levels differ at different points | [DMCurriculumHet](./api/dm.md#bolt.DMCurriculumHet) |
| High-dimensional | | [PO128](./api/po.md#bolt.PO128), [PO256](./api/po.md#bolt.PO256), [PO512](./api/po.md#bolt.PO512), [PO768](./api/po.md#bolt.PO768) |

---
 
## Common Interface
 
All problems follow botorch's BaseTestProblem interface.
 
```python
prob = HPO(noise_std=0.001, negate=False)
 
prob(X)                # returns objective value(s)
prob._bounds           # list of (min, max) per dimension
prob.dim               # total number of decision variables
prob.continuous_inds     # indices of continuous variables
prob.discrete_inds       # indices of integer/discrete variables
prob.categorical_inds    # indices of categorical variables

```

Type-specific attributes/functions are available depending on the problem:

| Attribute/Function | Type | Description |
|---|---|---|
| `prob.num_objectives` | Multi-objective | Number of objectives |
| `prob.cost(X)` | Multi-fidelity | Cost of querying a given fidelity at `X` |
| `prob.evaluate_noise(X)` | Heteroscedastic | Noise std at `X` |

---

## Observation Noise

The HPO and data mixture problems are **noisy by default**, at an empirically
measured level. `noise_std` defaults to each problem's `_measured_std` — the
standard deviation observed across repeat training runs of the real task — rather
than to `None`. This is the setting the benchmark results were produced under.

Set `noise_std` to override it — a value of your own for a different noise level,
or `None` for a noiseless problem. `_measured_std` only holds the measured value;
`noise_std` is the knob.

```python
prob = HPO()                    # noise_std = 0.0290, the measured value
prob = HPO(noise_std=None)      # noiseless
prob = HPO(noise_std=0.01)      # your own level

prob(X)                         # noisy
prob(X, noise=False)            # noiseless, whatever the default
```

| Problem | default `noise_std` | Empirically measured over |
|---|---|---|
| `HPO` | 0.0290 | 30 configs x 5 seeds, final checkpoint |
| `HPOMultiFidelityToken` | 0.0220 | 90 (config, checkpoint) groups x 5 seeds, all fidelities |
| `HPOMultiFidelityModel` | 0.0290 | 30 configs x 5 seeds, final checkpoint (8B; reused for 4B) |
| `DMCurriculum` | 0.0114 | 100 configs x 5 seeds |
| `DMCurriculumMO` | `[0.0131, 0.0274, 0.0101]` | 100 configs x 5 seeds, per objective (IFEval / MATH-500 / MBPP+) |

`DMCurriculumHet` is the exception: `noise_std` defaults to `None` and is ignored,
because its noise is input-dependent. Use `prob.evaluate_noise(X)` to read the std
at `X`.

Its objective is the mean of the same 3 benchmarks as `DMCurriculum`, but **only
MATH-500 has a noise emulator**. The IFEval and MBPP+ stds are far flatter over the
input space, so they are held at the constants measured for `DMCurriculumMO`:

| Benchmark | Noise | Value |
|---|---|---|
| IFEval | constant | 0.0131165 |
| MATH-500 | input-dependent | noise emulator evaluated at `X` |
| MBPP+ | constant | 0.0100820 |

The two constants are the `DMCurriculumMO` per-objective values unrounded. Replicate
deviations are near-uncorrelated across the three benchmarks, so the std of their
average adds in quadrature:

```text
evaluate_noise(X) = sqrt(sigma_if^2 + sigma_math(X)^2 + sigma_code^2) / 3
```

Note this departs from botorch, where `noise_std` defaults to `None`. Per-call
`prob(X, noise=False)` is unaffected and still returns the noiseless value.
 
---
 
## Problem Index
 
Full alphabetical listing of all problems. Click the name to jump to its API reference entry.

| Problem | Type(s) | Dim | Objectives | Description |
|---|---|---|---|---|
| [DMCurriculum](./api/dm.md#bolt.DMCurriculum) | Simplex constrained | 6 | 1 | Data mixture curriculum optimization (inputs must fulfill two simplex constraints) |
| [DMCurriculumHet](./api/dm.md#bolt.DMCurriculumHet) | Simplex constrained, heteroscedastic noise | 6 | 1 | Data mixture curriculum optimization with heteroscedastic noise |
| [DMCurriculumMO](./api/dm.md#bolt.DMCurriculumMO) | Simplex constrained, multi-objective | 6 | 3 | Data mixture curriculum optimization with multiple objectives |
| [HPO](./api/hpo.md#bolt.HPO) | Mixed-variable | 7 | 1 | Hyperparameter optimization for LoRA finetuning |
| [HPOMultiFidelityModel](./api/hpo.md#bolt.HPOMultiFidelityModel) | Mixed-variable, multi-fidelity | 8 | 1 | Hyperparameter optimization with fidelity controlled by model size |
| [HPOMultiFidelityToken](./api/hpo.md#bolt.HPOMultiFidelityToken) | Mixed-variable, multi-fidelity | 8 | 1 | Hyperparameter optimization with fidelity controlled by number of training tokens |
| [PO128](./api/po.md#bolt.PO128) | High-dimensional | 128 | 1 | Prompt optimization in high-dimensional discretized search space |
| [PO256](./api/po.md#bolt.PO256) | High-dimensional | 256 | 1 | Prompt optimization in high-dimensional discretized search space |
| [PO512](./api/po.md#bolt.PO512) | High-dimensional | 512 | 1 | Prompt optimization in high-dimensional discretized search space |
| [PO768](./api/po.md#bolt.PO768) | High-dimensional | 768 | 1 | Prompt optimization in high-dimensional discretized search space |

