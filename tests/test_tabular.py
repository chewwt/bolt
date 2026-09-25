import pytest
import torch

from bolt.functions.tabular import TabularFunction, TabularFunctionEmbeddings

HF_REPO = "chewwt/po_qwen14b_tabular_data"
EMB_COL = "embedding"
SCORE_COL = "score"
EMB_DIM = 768  # full embedding dimension in the dataset


def test_tabular_embeds() -> None:
    fn = TabularFunctionEmbeddings(HF_REPO, [EMB_COL], SCORE_COL)

    # test xs and ys shape
    assert fn.ys.shape[0] == fn.Xs.shape[0]
    n = len(fn.ys)
    assert fn.Xs.shape == (n, EMB_DIM)
    assert fn.ys.shape == (n,)

    # test evaluate_true output
    X = fn.Xs[:16]
    out = fn.evaluate_true(X)
    assert out.shape == (16, 1)

    expected = fn.ys[:16].unsqueeze(-1).to(dtype=out.dtype)
    torch.testing.assert_close(out, expected)

    # reject multiple input_cols for TabularFunctionEmbeddings
    with pytest.raises(AssertionError):
        TabularFunctionEmbeddings(HF_REPO, [EMB_COL, "other"], SCORE_COL)


PCO_REPO = "Glow-AI/pco32_tabular_data"
PCO_COLS = ["log2_dp_size", "log2_tp_size", "log2_pp_size"]


def test_tabular_multi_column() -> None:
    fn = TabularFunction(PCO_REPO, PCO_COLS, "throughput_mean", revision="v0.1.0")

    # one column per input, in input_cols order
    n = len(fn.ys)
    assert fn.Xs.shape == (n, len(PCO_COLS))
    for i, col in enumerate(PCO_COLS):
        torch.testing.assert_close(
            fn.Xs[:, i], torch.tensor(fn.ds[col], dtype=fn.Xs.dtype)
        )

    # OOM rows have null throughput, loaded as NaN rather than object dtype
    assert fn.ys.dtype == torch.float64
    assert fn.ys.isnan().any()

    # x_proc_func receives the stacked array
    fn_proc = TabularFunction(
        PCO_REPO,
        PCO_COLS,
        "throughput_mean",
        x_proc_func=lambda X: X * 2,
        revision="v0.1.0",
    )
    torch.testing.assert_close(fn_proc.Xs, fn.Xs * 2)
