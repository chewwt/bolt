from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# adapted from botorch testproblem


class Function(ABC, nn.Module):
    """Abstract base class for objective functions used in optimization problems.

    Subclasses must implement :meth:`_evaluate_true`, which is called by the
    public :meth:`evaluate_true` entry point.
    """

    def __init__(self) -> None:
        super().__init__()

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points.

        Args:
            X (torch.Tensor): A `(batch_shape) x d`-dim tensor of point(s) at which to
                evaluate.

        Returns:
            torch.Tensor: `batch_shape`-dim tensor of observations at corresponding
                points
        """

        out = self._evaluate_true(X)

        return out

    @abstractmethod
    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the function on a set of points (implementation hook).

        Args:
            X: A ``(batch_shape) x d``-dim tensor of input points.

        Returns:
            A ``batch_shape``-dim tensor of function values.
        """
        pass
