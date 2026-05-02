import pytest
import torch

from bolt.functions.tabular import TabularFunctionEmbeddings

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
