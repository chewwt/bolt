from abc import ABC, abstractmethod

import torch
from botorch.test_functions.base import BaseTestProblem, validate_inputs


class HeteroscedasticTestProblem(BaseTestProblem, ABC):
    r"""Base class for test functions with heteroscedastic noise."""

    def __init__(
        self,
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Base constructor for test functions.

        Args:
            noise_std: Standard deviation of the observation noise. Should not be set.
                Defaults to None. Kept for compatiblility with BaseTestProblem
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        if noise_std is not None:
            raise ValueError(
                f"noise_std for HeteroscedasticTestProblem should be None, currently it is set to {noise_std}"
            )

        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)

    def forward(self, X: torch.Tensor, noise: bool = True) -> torch.Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A ``(batch_shape) x d``-dim tensor of point(s) at which to evaluate
                the function.
            noise: If ``True``, add observation noise as specified by ``evaluate_noise``.

        Returns:
            A ``batch_shape``-dim tensor of function evaluations.
        """
        f = self.evaluate_true(X=X)

        if noise:
            _noise = self.evaluate_noise(X=X).to(device=X.device, dtype=X.dtype)
            f += _noise * torch.randn_like(f)
        if self.negate:
            f = -f
        return f

    def evaluate_noise(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate the noise function to get the noise_std on a set of points.

        Args:
            X: A ``(batch_shape) x d``-dim tensor of point(s) at which to
                evaluate the noise function.

        Returns:
            A ``batch_shape``-dim tensor.
        """
        validate_inputs(
            X=X,
            dim=self.dim,
            bounds=self.bounds,
            discrete_inds=self.discrete_inds,
            categorical_inds=self.categorical_inds,
        )
        return self._evaluate_noise(X=X)

    @abstractmethod
    def _evaluate_noise(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the noise function to get the noise_std on a set of points.

        Args:
            X: A ``(batch_shape) x d``-dim tensor of point(s) at which to
                evaluate the noise function.

        Returns:
            A ``batch_shape``-dim tensor.
        """
        pass  # pragma: no cover


class LLMTestProblem(BaseTestProblem, ABC):
    """Abstract base class for all BOLT LLM optimization problems.

    Subclasses must declare the class attributes ``name``, ``hf_repo``, ``dim``,
    ``_bounds``, and the index lists ``continuous_inds``, ``discrete_inds``,
    ``categorical_inds``, then implement :meth:`_evaluate_true`.

    This class is BoTorch-compatible: it extends
    :class:`botorch.test_functions.base.BaseTestProblem`.
    """

    name: str
    hf_repo: str  # must be set by subclass

    # params in BaseTestProblem
    dim: int  # defined in config.json on hf repo
    _bounds: list[
        tuple[float, float]
    ]  # Bounds, must be integers for discrete/categorical parameters
    _check_grad_at_opt: bool = True
    continuous_inds: list[int] = []
    discrete_inds: list[int] = []
    categorical_inds: list[int] = []
    _is_minimization_by_default: bool = True

    def __init__(
        self,
        # args in BaseTestProblem init
        noise_std: None | float | list[float] = None,
        negate: bool = False,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""Base class of a LLM optimization problem

        Args:
            noise_std: Standard deviation of the observation noise. If a list is
                provided, specifies separate noise standard deviations for each
                objective in a multiobjective problem.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """

        super().__init__(noise_std=noise_std, negate=negate, dtype=dtype)
