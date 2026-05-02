from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .base import Function

# feature net from bobench_data


class FeatureNet(nn.Module):
    r"""Three-layer MLP with LayerNorm for predicting LLM evaluation scores.

    Categorical parameters are one-hot encoded before being fed to the network.
    The network operates on the expanded *feature* representation; helper methods
    :meth:`proc_X_to_feat_X` and :meth:`proc_feat_X_to_X` convert between the
    compact raw-parameter space and the expanded feature space.
    """

    def __init__(
        self,
        input_dim: int,
        cat_indices: list[int] = list(),
        cat_nums: list[int] = list(),
        hidden_dim: int = 256,
        output_dim: int = 1,
    ):
        r"""
        Args:
            input_dim: Number of input features *after* one-hot expansion of categoricals.
            cat_indices: Positions of categorical columns in the raw (pre-expansion) input.
            cat_nums: Number of categories for each entry in ``cat_indices`` (same order).
            hidden_dim: Width of each hidden layer.
            output_dim: Number of network outputs.
        """
        super().__init__()

        # sort
        sort_inds = np.argsort(cat_indices)
        cat_indices = list(np.array(cat_indices)[sort_inds])
        cat_nums = list(np.array(cat_nums)[sort_inds])

        self.cat_indices = cat_indices
        self.cat_nums = cat_nums

        self.total_n_cat = np.sum(cat_nums)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(int(self.input_dim), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Run the MLP on pre-processed feature inputs.

        Args:
            X: Feature tensor of shape ``(N, input_dim)`` — already one-hot
               expanded. Use :meth:`get_pred` for raw parameter inputs.

        Returns:
            Output tensor of shape ``(N, output_dim)``.
        """
        feat_X = X.to(X.device)
        return self.mlp(feat_X)

    def proc_X_to_feat_X(self, X: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
        r"""Convert raw parameters to the expanded feature representation.

        Categorical columns are replaced by soft one-hot vectors (smoothed by
        ``eps`` to keep gradients non-zero). All other columns are passed through
        unchanged.

        Args:
            X: Raw input tensor of shape ``(N, d_raw)``.
            eps: Smoothing factor applied to one-hot encodings so gradients exist.

        Returns:
            Feature tensor of shape ``(N, input_dim)`` ready for :meth:`forward`.
        """
        # assemble input into mlp

        # split the tensors
        split_X = []
        start_i = 0
        for cat_i, cat_num in zip(self.cat_indices, self.cat_nums):
            if start_i != cat_i:
                # append non-cat Xs
                split_X.append(X[:, start_i:cat_i])

            # process cat Xs using one-hot (assume small no. of categories)
            one_hot = F.one_hot(X[:, cat_i].long(), num_classes=cat_num).float()
            one_hot = one_hot * (1 - eps) + eps / cat_num  # eps instead of 0 for grad
            split_X.append(one_hot)

            start_i = cat_i + 1

        # append last of non-cat Xs
        if start_i < X.shape[-1]:
            split_X.append(X[:, start_i:])

        # input X
        feat_X = torch.cat(split_X, dim=1)

        return feat_X

    def proc_feat_X_to_X(self, feat_X: torch.Tensor) -> torch.Tensor:
        r"""Invert :meth:`proc_X_to_feat_X`: recover raw parameters from feature tensor.

        One-hot blocks are collapsed back to integer category indices via argmax.

        Args:
            feat_X: Feature tensor of shape ``(N, input_dim)``.

        Returns:
            Raw parameter tensor of shape ``(N, d_raw)``.
        """
        split_X = []
        start_i = 0
        offset = 0

        for cat_i, cat_num in zip(self.cat_indices, self.cat_nums):
            if start_i + offset != cat_i + offset:
                # append non-cat Xs
                split_X.append(feat_X[:, (start_i + offset) : (cat_i + offset)])

            # recover category index from one hot encoding
            split_X.append(
                feat_X[:, (cat_i + offset) : (cat_i + offset + cat_num)].argmax(dim=-1)[
                    None, ...
                ]
            )

            start_i = cat_i
            offset += cat_num

        # append last of non-cat Xs
        if start_i + offset < feat_X.shape[-1]:
            split_X.append(feat_X[:, start_i + offset :])

        # back to original X
        X = torch.cat(split_X, dim=1)

        return X

    def get_pred(self, X: torch.Tensor) -> torch.Tensor:
        r"""Predict from raw parameters, handling one-hot encoding internally.

        Args:
            X: Raw input tensor of shape ``(N, d_raw)``.

        Returns:
            Detached output tensor of shape ``(N, output_dim)``.
        """

        feat_X = self.proc_X_to_feat_X(X).to(X.device)

        out = self.mlp(feat_X)

        return out.detach()


# Functions


class MLPFunction(Function):
    def __init__(
        self,
        input_dim: int,
        model_path: str,
        categorical_inds: list[int] = list(),
        categorical_sizes: list[int] = list(),
        hidden_dim: int = 256,
        output_dim: int = 1,
    ):
        r"""Load a pretrained :class:`FeatureNet` from disk.

        Args:
            input_dim: Number of raw input dimensions (before one-hot expansion).
            model_path: Absolute or relative path to the weights file
                (``.safetensors`` or ``.pth``).
            categorical_inds: Column indices of categorical parameters in the raw
                input (order does not matter; sorted internally).
            categorical_sizes: Number of categories for each entry in
                ``categorical_inds`` (must be in the same order).
            hidden_dim: Width of each hidden layer in the MLP.
            output_dim: Number of network outputs.
        """

        super().__init__()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Path to model {model_path} does not exists")

        self.dim = input_dim

        self.model = FeatureNet(
            input_dim,
            cat_indices=categorical_inds,
            cat_nums=categorical_sizes,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        if Path(model_path).suffix == ".safetensors":
            self.model.load_state_dict(load_file(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the MLP emulator on raw input parameters.

        Args:
            X: Raw input tensor of shape ``(N, input_dim)``.

        Returns:
            Output tensor of shape ``(N, output_dim)`` in the same dtype as ``X``.
        """
        # cast to model dtype temporarily to prevent type error
        model_dtype = next(self.model.parameters()).dtype
        pred = self.model.get_pred(X.to(dtype=model_dtype))
        return pred.to(dtype=X.dtype)
