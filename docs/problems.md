# Problems

A catalogue of LLM optimization problems are available in this library. 

Problems are backed by one or more pretrained emulators or tabular data downloaded automatically from HuggingFace Hub and cached locally for subsequent use. Emulator accuracy varies by problem — for Spearman's rank correlation coefficients and validation methodology see our [paper](#).
 
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
| `prob.n_objectives` | Multi-objective | Number of objectives |
| `prob.cost(X)` | Multi-fidelity | Cost of querying a given fidelity at `X` |
 
---
 
## Problem Index
 
Full alphabetical listing of all problems. Click the name to jump to its API reference entry.

| Problem | Type(s) | Dim | Objectives | Description |
|---|---|---|---|---|
| [DMCurriculum](./api/dm.md#bolt.DMCurriculum) | Simplex constrained | 6 | 1 | Data mixture curriculum optimization (inputs must fulfill two simplex contraints) |
| [DMCurriculumHet](./api/dm.md#bolt.DMCurriculumHet) | Simplex constrained, heteroscedastic noise | 6 | 1 | Data mixture curriculum optimization with heteroscedastic noise |
| [DMCurriculumMO](./api/dm.md#bolt.DMCurriculumMO) | Simplex constrained, multi-objective | 6 | 3 | Data mixture curriculum optimization with multiple objectives |
| [HPO](./api/hpo.md#bolt.HPO) | Mixed-variable | 7 | 1 | Hyperparameter optimization for LoRA finetuning |
| [HPOMultiFidelityModel](./api/hpo.md#bolt.HPOMultiFidelityModel) | Mixed-variable, multi-fidelity | 8 | 1 | Hyperparameter optimization with fidelity controlled by model size |
| [HPOMultiFidelityToken](./api/hpo.md#bolt.HPOMultiFidelityToken) | Mixed-variable, multi-fidelity | 8 | 1 | Hyperparameter optimization with fidelity controlled by number of training tokens |
| [PO128](./api/po.md#bolt.PO128) | High-dimensional | 128 | 1 | Prompt optimization in high-dimensional discretized search space |
| [PO256](./api/po.md#bolt.PO256) | High-dimensional | 256 | 1 | Prompt optimization in high-dimensional discretized search space |
| [PO512](./api/po.md#bolt.PO512) | High-dimensional | 512 | 1 | Prompt optimization in high-dimensional discretized search space |
| [PO768](./api/po.md#bolt.PO768) | High-dimensional | 768 | 1 | Prompt optimization in high-dimensional discretized search space |

