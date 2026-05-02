from pathlib import Path

import torch
from safetensors.torch import load_file

from .base import Function


class KernelRegressionModel(torch.nn.Module):
    r"""Non-parametric kernel regression model using an L1-distance exponential kernel.

    Predictions are computed as ``K(X, X_fit) @ dual_coef``, where
    ``K(a, b) = exp(-gamma * ||a - b||_1)``.
    """

    dual_coef: torch.Tensor
    X_fit: torch.Tensor

    def __init__(
        self, dual_coef: torch.Tensor, X_fit: torch.Tensor, gamma: float = 0.1
    ) -> None:
        r"""
        Args:
            dual_coef: Fitted dual coefficients of shape ``(n_train, n_outputs)``.
            X_fit: Training inputs used during fitting, shape ``(n_train, d)``.
            gamma: Bandwidth parameter of the exponential kernel.
        """
        super().__init__()
        self.gamma = gamma
        self.register_buffer("dual_coef", dual_coef)
        self.register_buffer("X_fit", X_fit)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Predict outputs for new inputs via kernel-weighted interpolation.

        Args:
            X: Input tensor of shape ``(N, d)``.

        Returns:
            Prediction tensor of shape ``(N, n_outputs)``.
        """
        dist = torch.cdist(X, self.X_fit, p=1)
        K = torch.exp(-self.gamma * dist)
        return K @ self.dual_coef


# Functions


class KRFunction(Function):
    r"""Function to call KernelRegressionModel"""

    def __init__(self, model_path: str, scale_factor: float = 1.0):
        r"""Load a pretrained :class:`KernelRegressionModel` from a ``.safetensors`` file.

        Args:
            model_path: Path to the ``.safetensors`` file containing
                ``dual_coef`` and ``X_fit`` tensors.
            scale_factor: Scalar multiplier applied to all predictions (useful
                for re-scaling noise estimates).
        """

        super().__init__()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Path to model {model_path} does not exists")

        tensors = load_file(model_path)
        self.model = KernelRegressionModel(tensors["dual_coef"], tensors["X_fit"])
        self.model.eval()

        self.scale_factor = scale_factor

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the kernel regression model on input points.

        Args:
            X: Input tensor of shape ``(N, d)``.

        Returns:
            Scaled prediction tensor of shape ``(N, n_outputs)`` in the same
            dtype as ``X``.
        """
        # cast to model dtype temporarily to prevent type error
        model_dtype = self.model.dual_coef.dtype
        pred = self.scale_factor * self.model(X.to(dtype=model_dtype))

        return pred.to(dtype=X.dtype)
