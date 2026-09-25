import warnings
from typing import Optional

import numpy as np
import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem

from ..functions.tabular import TabularFunction
from .base import LLMTestProblem

# Per-GPU HBM the PCO32/PCO64 sweeps ran on; an instance on other hardware
# overrides `PCO.mem_capacity_gb`. The raw sweeps record
# `peak_mem / mem_capacity_gb - 1`, so the margin is negative exactly while the
# run fits -- BoTorch's `c(x) <= 0` convention.
MEM_CAPACITY_GB = 96.0

# Margin recorded in place of a measurement for a run that died: an OOM kills
# the process before it reports memory, so the overflow has no magnitude and the
# sweep pads it with a fixed positive value.
OOM_MARGIN_PAD = 0.2

# Two neighbours count as equidistant within this tolerance. Distances here are
# sums of squared integer log2 steps, so anything below 1 would do.
TIE_TOL = 1e-9

# Beyond this distance a query is not a row. Rows are a log2 step apart.
SNAP_TOL = 1e-8

# The table stores the raw ZeRO stage; it is searched as an ordinal over these
# levels, monotone in how aggressively optimizer state is sharded.
ZERO_STAGE_LEVELS = (0, 2, 3)


def _encode_zero_stage(X: np.ndarray, col: int) -> np.ndarray:
    r"""Map the raw ``zero_stage`` column of a table to its ordinal index.

    Args:
        X: Input table of shape ``(table_size, d)``.
        col: Index of the ``zero_stage`` column.

    Returns:
        A copy of ``X`` with column ``col`` replaced by its index into
        ``ZERO_STAGE_LEVELS``.

    Raises:
        ValueError: If the column holds a stage outside ``ZERO_STAGE_LEVELS``.
    """
    X = X.copy()
    levels = np.asarray(ZERO_STAGE_LEVELS, dtype=X.dtype)
    idx = np.searchsorted(levels, X[:, col])
    if not np.array_equal(levels[np.minimum(idx, len(levels) - 1)], X[:, col]):
        raise ValueError(
            f"zero_stage outside {ZERO_STAGE_LEVELS}: {sorted(set(X[:, col]))}"
        )
    X[:, col] = idx
    return X


class PCO(LLMTestProblem, ConstrainedBaseTestProblem):
    r"""Abstract base for parallelism configuration optimization problems.

    Choose a parallelism configuration for pre-training a dense transformer on a
    fixed cluster, maximising training throughput without running out of GPU
    memory. Subclasses fix the model shape and cluster size and set ``hf_repo``,
    ``_bounds`` and the reference optimum.

    The search space is the finite set of configurations in the table: the
    parallelism degrees must multiply to the GPU count, so most of the bounding
    box is not a valid configuration at all. Drive it with
    ``optimize_acqf_discrete`` over :meth:`candidates` rather than over the box.

    Every parameter is log2 of the underlying size, except ``zero_stage``, which
    is an ordinal index over (0, 2, 3). Each knob halves or doubles, so log2
    makes the levels evenly spaced for a stationary kernel.

    8 parameters (PCO32/PCO64; PCO16 differs, see :class:`PCO16`):
        1. log2 data-parallel size (int)
        2. log2 tensor-parallel size (int)
        3. log2 pipeline-parallel size (int)
        4. log2 context-parallel size (int)
        5. log2 sequence-parallel size (int)
        6. log2 DDP gradient bucket size in MB (int)
        7. ZeRO stage, ordinal over (0, 2, 3) (int)
        8. log2 number of model chunks (int)

    Objective: mean training throughput (iterations/second), to be maximised.

    Constraint: the run must not exhaust GPU memory, i.e. peak memory must stay
    under ``mem_capacity_gb``. The slack is the run's own memory margin,
    continuous and informative on the feasible side; where the run OOMed it is
    the fixed pad ``OOM_MARGIN_PAD``, because the process dies before reporting
    how far over it went.

    This is a *black-box* constraint, and the main challenge of the problem:
    feasibility has no closed form and is only revealed by running the
    configuration. An infeasible configuration OOMs, so its throughput is never
    measured. :meth:`evaluate_true` fills the gap with the
    mean throughput of the equidistant nearest feasible neighbours, keeping the
    objective surface smooth across the boundary. It is done here, not per
    optimizer, so every method sees the same numbers.

    Incumbents and regret must count feasible points only (see
    :meth:`is_feasible`): the imputed value at an OOM is a throughput no run
    achieved.

    Because feasibility is only revealed by attempting the run, the objective
    and the constraint cannot be evaluated separately -- the problem is coupled.
    """

    name: str
    hf_repo: str
    hf_revision: Optional[str] = "v0.1.0"

    # Constants of the sweep, not search dimensions: they are the same in every
    # row, so they are not columns of `X`. A prior mean over the cost of a
    # configuration needs them, since the cluster shape is what the parallelism
    # degrees are being fitted to.
    num_gpus: int
    num_hosts: int
    batch_size: int
    mem_capacity_gb: float = MEM_CAPACITY_GB

    # Input columns of the tabular dataset, in parameter order. All are already
    # the search encoding except `zero_stage`, which is raw (see
    # `_encode_zero_stage`).
    input_cols = [
        "log2_dp_size",
        "log2_tp_size",
        "log2_pp_size",
        "log2_cp_size",
        "log2_sp_size",
        "log2_dp_bucket_size_mb",
        "zero_stage",
        "log2_num_model_chunks",
    ]
    objective_col = "throughput_mean"

    dim = 8
    num_constraints = 1
    # Piecewise-constant lookup, no gradient to check.
    _check_grad_at_opt: bool = False
    continuous_inds = []
    discrete_inds = list(range(8))
    categorical_inds = []

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Parallelism configuration optimization.

        Args:
            noise_std: Standard deviation of the observation noise. Defaults to
                ``None``, i.e. noiseless observations. The table holds one run
                per configuration, so there is no measured seed-to-seed spread
                to default to.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

        zero_stage_col = self.input_cols.index("zero_stage")
        self.obj_func = TabularFunction(
            self.hf_repo,
            self.input_cols,
            self.objective_col,
            x_proc_func=lambda X: _encode_zero_stage(X, zero_stage_col),
            revision=self.hf_revision,
        )

        # From the same table as obj_func, so rows stay aligned with `Xs`. A run
        # that OOMed records no throughput, so feasibility is a finite objective.
        ds = self.obj_func.ds
        self.register_buffer("feasible", torch.isfinite(self.obj_func.ys))
        self.register_buffer(
            "peak_mem",
            torch.tensor(np.asarray(ds["peak_mem"], dtype=np.float64), dtype=dtype),
        )

        self.register_buffer("y_imputed", self._impute_table(dtype))
        self.register_buffer("margin", self._memory_margin(dtype))

    def _memory_margin(self, dtype: torch.dtype) -> torch.Tensor:
        r"""Per-row memory margin, negative iff the configuration fits.

        The fraction of the GPU's capacity over budget, padded to
        ``OOM_MARGIN_PAD`` where the run OOMed and ``peak_mem`` is ``inf``.

        Args:
            dtype: dtype of the returned tensor.

        Returns:
            torch.Tensor: ``(table_size,)``-dim tensor of margins.
        """
        margin = self.peak_mem.to(dtype=dtype) / self.mem_capacity_gb - 1.0
        return torch.where(
            self.feasible, margin, torch.full_like(margin, OOM_MARGIN_PAD)
        )

    def _impute_table(self, dtype: torch.dtype) -> torch.Tensor:
        r"""Objective table with each OOM row filled from its nearest neighbours.

        The fill is the mean over every feasible row at the minimum distance:
        most OOM rows tie several neighbours a log2 step away, so a single
        ``argmin`` would pick among them by index order.

        Args:
            dtype: dtype of the returned tensor.

        Returns:
            torch.Tensor: ``(table_size,)``-dim tensor of throughputs, with no
            NaN unless the table has no feasible row at all.
        """
        y = self.obj_func.ys.to(dtype=dtype).clone()
        feasible = self.feasible
        if not bool(feasible.any()) or bool(feasible.all()):
            return y

        Xs = self.obj_func.Xs.to(dtype=dtype)
        dists = torch.cdist(Xs[~feasible], Xs[feasible])
        ties = dists <= dists.min(dim=1, keepdim=True).values + TIE_TOL
        weights = ties.to(dtype=dtype)
        y[~feasible] = (weights @ y[feasible]) / weights.sum(dim=1)
        return y

    def candidates(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""The full candidate set: every configuration in the table.

        Args:
            dtype: dtype of the returned tensor. Defaults to the bounds' dtype.
            device: device of the returned tensor.

        Returns:
            torch.Tensor: ``(table_size, dim)``-dim tensor of encoded
            configurations.
        """
        dtype = self.bounds.dtype if dtype is None else dtype
        return self.obj_func.Xs.to(dtype=dtype, device=device)

    def _row_indices(self, X: torch.Tensor) -> torch.Tensor:
        r"""Index of the table row nearest each query point.

        A point that is not a row is snapped to the nearest one, with a warning:
        most of the bounding box is not a valid configuration, so a search over
        the box would otherwise get throughputs for configurations that cannot
        run.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: ``(N,)``-dim tensor of row indices.
        """
        Xs = self.obj_func.Xs.to(device=X.device, dtype=X.dtype)
        nearest = torch.cdist(X, Xs).min(dim=1)
        off_table = nearest.values > SNAP_TOL
        if bool(off_table.any()):
            n = int(off_table.sum())
            warnings.warn(
                f"{self.name}: {n}/{len(X)} points are not in the table; "
                "snapped to the nearest row. Search over `candidates()`.",
                RuntimeWarning,
                stacklevel=3,
            )
        return nearest.indices.cpu()

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Throughput of the nearest configuration in the table.

        Where that configuration ran out of memory the throughput was never
        measured, so it is imputed from the nearest feasible neighbours.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Objective tensor of shape ``(N, 1)``.
        """
        idx = self._row_indices(X)
        return self.y_imputed[idx].to(device=X.device, dtype=X.dtype).unsqueeze(-1)

    def _evaluate_slack_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Feasibility slack, positive iff the configuration fits in memory.

        The negated memory margin: the fraction of GPU capacity still free, and
        the constant ``-OOM_MARGIN_PAD`` where the run died, since an OOM reports
        no memory figure. Those entries are censored (all that is known is
        ``slack < 0``), not measurements of the overflow.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Slack tensor of shape ``(N, 1)``.
        """
        idx = self._row_indices(X)
        slack = -self.margin[idx].to(device=X.device, dtype=X.dtype)
        return slack.unsqueeze(-1)

    def evaluate_peak_mem(self, X: torch.Tensor) -> torch.Tensor:
        r"""Peak GPU memory of the nearest configuration, in GB.

        ``inf`` where the run OOMed. This is an auxiliary observation, not the
        constraint -- it is the signal a prior mean function can be built on.

        Args:
            X (torch.Tensor): Input tensor of shape ``(N, dim)``.

        Returns:
            torch.Tensor: Memory tensor of shape ``(N, 1)``.
        """
        idx = self._row_indices(X)
        return self.peak_mem[idx].to(device=X.device, dtype=X.dtype).unsqueeze(-1)


class PCO16(PCO):
    r"""Parallelism configuration for a 32-layer model on 16 GPUs (2 hosts).

    Model: 32 layers, 32 attention heads, hidden size 4096, FFN hidden size
    11008, batch size 512, sequence length 2048.

    Unlike PCO32/PCO64, this sweep varies gradient accumulation and fixes
    context parallelism at 1, so ``log2_grad_accum_steps`` replaces
    ``log2_cp_size``; it is the eighth parameter.

    919 configurations, of which 632 fit in memory.

    Example usage:
    ```python
    from bolt import PCO16

    prob = PCO16()
    X = prob.candidates()[:4]
    y = prob(X)  # NN-imputed where the configuration OOMs
    slack = prob.evaluate_slack(X)  # memory margin, > 0 iff feasible
    feasible = prob.is_feasible(X)  # count only these in regret
    ```
    """

    name = "pco16"
    hf_repo = "Glow-AI/pco16_tabular_data"

    num_gpus = 16
    num_hosts = 2
    batch_size = 512
    # Swept on 80 GB GPUs, not the 96 GB of PCO32/PCO64.
    mem_capacity_gb = 80.0

    input_cols = [
        "log2_dp_size",
        "log2_tp_size",
        "log2_pp_size",
        "log2_sp_size",
        "log2_dp_bucket_size_mb",
        "zero_stage",
        "log2_num_model_chunks",
        "log2_grad_accum_steps",
    ]

    _bounds = [
        (1, 4),  # log2_dp_size
        (0, 3),  # log2_tp_size
        (0, 3),  # log2_pp_size
        (0, 3),  # log2_sp_size
        (6, 9),  # log2_dp_bucket_size_mb
        (0, 2),  # zero_stage (ordinal)
        (0, 2),  # log2_num_model_chunks
        (0, 3),  # log2_grad_accum_steps
    ]

    # Best feasible row in the table.
    _optimal_value = 0.1489374720046964
    # raw: dp 16, tp 1, pp 1, sp 1, bucket 256 MB, ZeRO 2, chunks 1, grad accum 4
    _optimizers = [(4, 0, 0, 0, 8, 1, 0, 2)]


class PCO32(PCO):
    r"""Parallelism configuration for a 40-layer model on 32 GPUs (8 hosts).

    Model: 40 layers, 40 attention heads, hidden size 5120, FFN hidden size
    17408, batch size 512, sequence length 8192.

    379 configurations, of which 339 fit in memory.

    Example usage:
    ```python
    from bolt import PCO32

    prob = PCO32()
    X = prob.candidates()[:4]
    y = prob(X)  # NN-imputed where the configuration OOMs
    slack = prob.evaluate_slack(X)  # memory margin, > 0 iff feasible
    feasible = prob.is_feasible(X)  # count only these in regret
    ```
    """

    name = "pco32"
    hf_repo = "Glow-AI/pco32_tabular_data"

    num_gpus = 32
    num_hosts = 8
    batch_size = 512

    _bounds = [
        (1, 5),  # log2_dp_size
        (0, 2),  # log2_tp_size
        (0, 3),  # log2_pp_size
        (0, 2),  # log2_cp_size
        (0, 2),  # log2_sp_size
        (6, 9),  # log2_dp_bucket_size_mb
        (0, 2),  # zero_stage (ordinal)
        (0, 1),  # log2_num_model_chunks
    ]

    # Best feasible row in the table.
    _optimal_value = 0.469575911560675
    # raw: dp 32, tp 1, pp 1, cp 1, sp 1, bucket 256 MB, ZeRO 2, chunks 1
    _optimizers = [(5, 0, 0, 0, 0, 8, 1, 0)]


class PCO64(PCO):
    r"""Parallelism configuration for a 64-layer model on 64 GPUs (16 hosts).

    Model: 64 layers, 64 attention heads, hidden size 5120, FFN hidden size
    25600, batch size 512, sequence length 8192.

    780 configurations, of which 560 fit in memory.

    Example usage:
    ```python
    from bolt import PCO64

    prob = PCO64()
    X = prob.candidates()[:4]
    y = prob(X)  # NN-imputed where the configuration OOMs
    slack = prob.evaluate_slack(X)  # memory margin, > 0 iff feasible
    feasible = prob.is_feasible(X)  # count only these in regret
    ```
    """

    name = "pco64"
    hf_repo = "Glow-AI/pco64_tabular_data"

    num_gpus = 64
    num_hosts = 16
    batch_size = 512

    _bounds = [
        (1, 6),  # log2_dp_size
        (0, 2),  # log2_tp_size
        (0, 4),  # log2_pp_size
        (0, 2),  # log2_cp_size
        (0, 2),  # log2_sp_size
        (6, 9),  # log2_dp_bucket_size_mb
        (0, 2),  # zero_stage (ordinal)
        (0, 2),  # log2_num_model_chunks
    ]

    # Best feasible row in the table.
    _optimal_value = 0.270684453437186
    # raw: dp 16, tp 4, pp 1, cp 1, sp 4, bucket 64 MB, ZeRO 2, chunks 1
    _optimizers = [(4, 2, 0, 0, 2, 6, 1, 0)]
