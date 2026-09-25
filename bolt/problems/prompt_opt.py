import warnings
from typing import Optional

import numpy as np
import torch

from ..functions.tabular import TabularFunctionEmbeddings
from .base import LLMTestProblem

# Beyond this distance a query is not a candidate. Well above float32 rounding of
# a candidate (~1e-7) and well below the gap between distinct prompts (median ~0.4).
SNAP_TOL = 1e-4


class PO(LLMTestProblem):
    r"""Abstract base for prompt optimization problems.

    Subclasses must define `dim` (int) and `name` (str) as class attributes.
    `_bounds` and `continuous_inds` are derived automatically from `dim`.

    Reward: Math500 0-shot score for Qwen3-14B given a prompt in the system instruction.
    Search space: `dim`-dimensional embeddings from EmbeddingGemma, bounded to (-0.20, 0.27).
    """

    hf_repo = "chewwt/po_qwen14b_tabular_data"
    hf_revision = "v0.1.0"

    _bounds_range: tuple[float, float] = (-0.20, 0.27)
    _check_grad_at_opt: bool = True
    _optimal_value = 0.81

    # Subclasses set this; __init_subclass__ derives _bounds and continuous_inds from it.
    dim: int

    def __init_subclass__(cls, **kwargs):
        """Automatically derive ``_bounds`` and ``continuous_inds`` from ``dim``."""
        super().__init_subclass__(**kwargs)
        if "dim" in cls.__dict__:
            d = cls.__dict__["dim"]
            cls._bounds = [cls._bounds_range] * d
            cls.continuous_inds = list(range(d))

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Prompt optimization for Qwen3-14B on MATH500.

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if not hasattr(self, "_bounds"):
            raise TypeError(
                f"{type(self).__name__} must define `dim` as a class attribute."
            )

        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

        self.obj_func = TabularFunctionEmbeddings(
            self.hf_repo,
            ["embedding"],
            "score",
            x_proc_func=lambda x: np.vstack(x)[:, : self.dim],
            revision=self.hf_revision,
        )

    def candidates(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        r"""The full candidate set: every prompt embedding in the table.

        Args:
            dtype: dtype of the returned tensor. Defaults to the bounds' dtype.
            device: device of the returned tensor.

        Returns:
            torch.Tensor: ``(table_size, dim)``-dim tensor of embeddings.
        """
        dtype = self.bounds.dtype if dtype is None else dtype
        return self.obj_func.Xs.to(dtype=dtype, device=device)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the nearest neighbour to tabular data.

        A point that is not a candidate is snapped to the nearest one, with a
        warning.

        Args:
            X (torch.Tensor): Input tensor of shape `(N, dim)`.

        Returns:
            torch.Tensor: Objective tensor of shape `(N, 1)`.
        """
        Xs = self.obj_func.Xs.to(device=X.device, dtype=X.dtype)
        idx = torch.cdist(X, Xs).argmin(dim=1)
        # cdist's matmul shortcut is inexact (~1e-3 in float32), so measure the
        # distance to the chosen candidate directly.
        off_table = (X - Xs[idx]).norm(dim=-1) > SNAP_TOL
        if bool(off_table.any()):
            n = int(off_table.sum())
            warnings.warn(
                f"{type(self).__name__}: {n}/{len(X)} points are not candidates; "
                "snapped to the nearest one. Search over `candidates()`.",
                RuntimeWarning,
                stacklevel=3,
            )
        return (
            self.obj_func.ys[idx.cpu()].unsqueeze(-1).to(device=X.device, dtype=X.dtype)
        )


class PO128(PO):
    r"""Prompt optimization via search in a 128-dimensional embedding space.

    The search space consists of 128-dimensional truncated embeddings from EmbeddingGemma.
    The full discrete candidate set is returned by ``prob.candidates()``, and
    can be used directly for discrete optimization. Evaluating any point X via
    ``prob(X)`` returns the Math500 0-shot accuracy of its nearest neighbor in
    the candidate set.

    Example usage:
    ```python
    import torch
    from bolt import PO128

    prob = PO128(noise_std=0.001, negate=False)
    X = torch.zeros(1, prob.dim, dtype=torch.double)
    y = prob(X)
    ```
    """

    name = "po128"
    dim = 128


class PO256(PO):
    r"""Prompt optimization via search in a 256-dimensional embedding space.

    The search space consists of 256-dimensional truncated embeddings from EmbeddingGemma.
    The full discrete candidate set is returned by ``prob.candidates()``, and
    can be used directly for discrete optimization. Evaluating any point X via
    ``prob(X)`` returns the Math500 0-shot accuracy of its nearest neighbor in
    the candidate set.

    Example usage:
    ```python
    import torch
    from bolt import PO256

    prob = PO256(noise_std=0.001, negate=False)
    X = torch.zeros(1, prob.dim, dtype=torch.double)
    y = prob(X)
    ```
    """

    name = "po256"
    dim = 256


class PO512(PO):
    r"""Prompt optimization via search in a 512-dimensional embedding space.

    The search space consists of 512-dimensional truncated embeddings from EmbeddingGemma.
    The full discrete candidate set is returned by ``prob.candidates()``, and
    can be used directly for discrete optimization. Evaluating any point X via
    ``prob(X)`` returns the Math500 0-shot accuracy of its nearest neighbor in
    the candidate set.

    Example usage:
    ```python
    import torch
    from bolt import PO512

    prob = PO512(noise_std=0.001, negate=False)
    X = torch.zeros(1, prob.dim, dtype=torch.double)
    y = prob(X)
    ```
    """

    name = "po512"
    dim = 512


class PO768(PO):
    r"""Prompt optimization via search in a 768-dimensional embedding space.

    The search space consists of 768-dimensional truncated embeddings from EmbeddingGemma.
    The full discrete candidate set is returned by ``prob.candidates()``, and
    can be used directly for discrete optimization. Evaluating any point X via
    ``prob(X)`` returns the Math500 0-shot accuracy of its nearest neighbor in
    the candidate set.

    Example usage:
    ```python
    import torch
    from bolt import PO768

    prob = PO768(noise_std=0.001, negate=False)
    X = torch.zeros(1, prob.dim, dtype=torch.double)
    y = prob(X)
    ```
    """

    name = "po768"
    dim = 768
