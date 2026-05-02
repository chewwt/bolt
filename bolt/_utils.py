import json
from typing import Union

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


def unstandardize_y(
    y: torch.Tensor,
    y_mean: Union[np.ndarray, torch.Tensor],
    y_std: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    r"""Reverse standardization: map y from zero-mean/unit-variance back to original scale.

    Args:
        y: Standardized output tensor.
        y_mean: Per-output mean used during standardization.
        y_std: Per-output standard deviation used during standardization.

    Returns:
        Tensor of the same shape as ``y`` in the original (un-standardized) scale.
    """
    # y_mean/y_std may arrive as numpy arrays on CPU; move to y's device to avoid device mismatch
    y_mean = torch.as_tensor(y_mean, dtype=y.dtype, device=y.device)
    y_std = torch.as_tensor(y_std, dtype=y.dtype, device=y.device)
    return (y * y_std) + y_mean


def pull_info_from_hf_hub(hf_repo: str) -> tuple[str, dict, np.ndarray, np.ndarray]:
    r"""Download model weights, config, and standardization stats from a HuggingFace repo.

    Files fetched (cached locally by ``huggingface_hub``):
        - ``model.safetensors`` — pretrained weights
        - ``config.json`` — model architecture config
        - ``model_standardize.csv`` — per-output ``y_mean`` / ``y_std`` columns

    Args:
        hf_repo: HuggingFace repository id, e.g. ``"chewwt/hpo_qwen8b_emulator"``.

    Returns:
        A four-tuple ``(model_path, model_config, y_mean, y_std)`` where
        ``model_path`` is the local path to the weights file, ``model_config``
        is the parsed JSON dict, and ``y_mean`` / ``y_std`` are 1-D numpy arrays.
    """
    model_path = hf_hub_download(hf_repo, "model.safetensors")
    config_path = hf_hub_download(hf_repo, "config.json")
    csv_path = hf_hub_download(hf_repo, "model_standardize.csv")

    # read model config
    with open(config_path, "r") as f:
        model_config = json.load(f)

    # read standardization params
    df = pd.read_csv(csv_path)
    y_mean = df["y_mean"].to_numpy()
    y_std = df["y_std"].to_numpy()

    return model_path, model_config, y_mean, y_std
