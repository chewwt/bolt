from abc import ABC
from typing import Any, Callable, Optional

import numpy as np
import torch
from datasets import load_dataset

from .base import Function


class TabularFunction(Function, ABC):
    r"""Nearest-neighbor lookup against a preloaded table from HuggingFace via brute-force L2 search."""

    def __init__(
        self,
        hf_repo: str,
        input_cols: list[str],
        output_col: str,
        x_proc_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        r"""Load a tabular dataset from HuggingFace and build the lookup table.

        Args:
            hf_repo: HuggingFace dataset repository id.
            input_cols: Column name(s) used as the input features for NN search.
            output_col: Column name whose values are returned as function outputs.
            x_proc_func: Optional callable to transform the raw column data
                before converting to a tensor. Receives the column dict and
                must return an array-like.
        """
        super().__init__()

        self.ds = load_dataset(hf_repo, split="train")

        self.input_cols = input_cols
        self.output_col = output_col
        self.x_proc_func = x_proc_func

        y_np = np.array(self.ds[output_col])
        self.ys = torch.tensor(y_np)

        self.Xs = self.init_Xs(input_cols, x_proc_func)

    def init_Xs(
        self, input_cols: list[str], x_proc_func: Optional[Callable[..., Any]] = None
    ) -> torch.Tensor:
        r"""Build the input lookup table from the loaded dataset.

        Args:
            input_cols: Column name(s) to read from the dataset.
            x_proc_func: Optional transform applied to the raw column data.

        Returns:
            Float tensor of shape ``(table_size, d)`` used for nearest-neighbour
            search in :meth:`_evaluate_true`.
        """
        X_raw = self.ds[input_cols]
        X_np = np.asarray(x_proc_func(X_raw) if x_proc_func is not None else X_raw)
        return torch.tensor(X_np)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Look up the nearest neighbour in the table and return the corresponding output.

        Args:
            X: Query tensor of shape ``(N, d)``.

        Returns:
            Output tensor of shape ``(N, 1)`` containing the ``output_col`` value
            of the nearest table entry (by L2 distance).
        """
        # Brute-force exact L2 nearest-neighbor search — fast enough for table sizes ~5k
        dists = torch.cdist(
            X, self.Xs.to(device=X.device, dtype=X.dtype)
        )  # (N, table_size)
        idx = dists.argmin(dim=1).cpu()
        return self.ys[idx].unsqueeze(-1).to(device=X.device, dtype=X.dtype)


class TabularFunctionEmbeddings(TabularFunction):
    r"""Exact nearest-neighbor lookup for embedding column via brute-force L2 search.

    Embedding column contains lists to be stacked
    """

    def __init__(
        self,
        hf_repo: str,
        input_cols: list[str],
        output_col: str,
        x_proc_func: Optional[Callable[..., Any]] = None,
    ) -> None:
        r"""Load a tabular dataset where inputs are stored as embedding lists.

        Args:
            hf_repo: HuggingFace dataset repository id.
            input_cols: Exactly one column name whose entries are embedding lists.
            output_col: Column name whose values are returned as function outputs.
            x_proc_func: Optional callable to transform the raw list-of-embeddings
                before stacking into a tensor.
        """
        assert len(input_cols) == 1, (
            "TabularFunctionEmbeddings requires exactly one input column"
        )
        super().__init__(hf_repo, input_cols, output_col, x_proc_func=x_proc_func)

    def init_Xs(
        self, input_cols: list[str], x_proc_func: Optional[Callable[..., Any]] = None
    ) -> torch.Tensor:
        r"""Build the input lookup table by stacking the embedding lists.

        Args:
            input_cols: Single-element list with the embedding column name.
            x_proc_func: Optional transform applied to the raw list of embeddings.

        Returns:
            Float tensor of shape ``(table_size, embedding_dim)``.
        """
        X_raw = self.ds[input_cols[0]]
        X_np = np.asarray(x_proc_func(X_raw) if x_proc_func is not None else X_raw)
        return torch.tensor(X_np)
