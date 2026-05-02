import numpy as np
import torch

from ..functions.tabular import TabularFunctionEmbeddings
from .base import LLMTestProblem


class PO(LLMTestProblem):
    r"""Abstract base for prompt optimization problems.

    Subclasses must define `dim` (int) and `name` (str) as class attributes.
    `_bounds` and `continuous_inds` are derived automatically from `dim`.

    Reward: Math500 0-shot score for Qwen3-14B given a prompt in the system instruction.
    Search space: `dim`-dimensional embeddings from EmbeddingGemma, bounded to (-0.20, 0.27).
    """

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
            "chewwt/po_qwen14b_tabular_data",
            ["embedding"],
            "score",
            x_proc_func=lambda x: np.vstack(x)[:, : self.dim],
        )

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the objective using the nearest neighbour to tabular data.

        Args:
            X (torch.Tensor): Input tensor of shape `(N, dim)`.

        Returns:
            torch.Tensor: Objective tensor of shape `(N, 1)`.
        """
        return self.obj_func.evaluate_true(X)


class PO128(PO):
    r"""Prompt optimization via search in a 128-dimensional embedding space.

    The search space consists of 128-dimensional truncated embeddings from EmbeddingGemma.
    The full discrete candidate set is accessible at ``prob.obj_func.Xs``, and
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
    The full discrete candidate set is accessible at ``prob.obj_func.Xs``, and
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
    The full discrete candidate set is accessible at ``prob.obj_func.Xs``, and
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
    The full discrete candidate set is accessible at ``prob.obj_func.Xs``, and
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
