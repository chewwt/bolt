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
        revision: Optional[str] = None,
    ) -> None:
        r"""Load a tabular dataset from HuggingFace and build the lookup table.

        Args:
            hf_repo: HuggingFace dataset repository id.
            input_cols: Column name(s) used as the input features for NN search.
            output_col: Column name whose values are returned as function outputs.
            x_proc_func: Optional callable to transform the input table before
                converting to a tensor. Receives an ``(table_size, d)`` float64
                array with columns in ``input_cols`` order and must return an
                array-like.
            revision: Git revision to fetch — a tag, branch, or commit SHA.
                Defaults to the repo's main branch.
        """
        super().__init__()

        self.ds = load_dataset(hf_repo, split="train", revision=revision)

        self.input_cols = input_cols
        self.output_col = output_col
        self.x_proc_func = x_proc_func

        y_np = np.asarray(self.ds[output_col])
        if y_np.dtype == object:
            # `datasets` returns None for each null in a column, i.e. object
            # dtype, which torch cannot take. Coerce so the gaps become NaN.
            y_np = y_np.astype(np.float64)
        self.ys = torch.tensor(y_np)

        self.Xs = self.init_Xs(input_cols, x_proc_func)

    def init_Xs(
        self, input_cols: list[str], x_proc_func: Optional[Callable[..., Any]] = None
    ) -> torch.Tensor:
        r"""Build the input lookup table from the loaded dataset.

        Args:
            input_cols: Column name(s) to read from the dataset.
            x_proc_func: Optional transform applied to the stacked input table.

        Returns:
            Float tensor of shape ``(table_size, d)`` used for nearest-neighbour
            search in :meth:`_evaluate_true`.
        """
        # Indexing a `datasets.Dataset` with a list selects rows, not columns, so
        # the table is assembled one named column at a time.
        X_np = np.column_stack(
            [np.asarray(self.ds[col], dtype=np.float64) for col in input_cols]
        )
        if x_proc_func is not None:
            X_np = np.asarray(x_proc_func(X_np))
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
        revision: Optional[str] = None,
    ) -> None:
        r"""Load a tabular dataset where inputs are stored as embedding lists.

        Args:
            hf_repo: HuggingFace dataset repository id.
            input_cols: Exactly one column name whose entries are embedding lists.
            output_col: Column name whose values are returned as function outputs.
            x_proc_func: Optional callable to transform the raw list-of-embeddings
                before stacking into a tensor.
            revision: Git revision to fetch — a tag, branch, or commit SHA.
                Defaults to the repo's main branch.
        """
        assert len(input_cols) == 1, (
            "TabularFunctionEmbeddings requires exactly one input column"
        )
        super().__init__(
            hf_repo,
            input_cols,
            output_col,
            x_proc_func=x_proc_func,
            revision=revision,
        )

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
