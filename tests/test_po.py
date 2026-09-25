import warnings

import pytest
import torch

from bolt import PO128, PO256, PO512, PO768
from bolt.problems.prompt_opt import PO


@pytest.mark.parametrize(
    "cls,dim", [(PO128, 128), (PO256, 256), (PO512, 512), (PO768, 768)]
)
def test_po_output_shape(cls: PO, dim: int) -> None:
    prob = cls(noise_std=0.001, negate=False)
    assert prob.dim == dim
    assert len(prob._bounds) == dim
    assert len(prob.continuous_inds) == dim

    X = torch.zeros((4, dim), dtype=torch.double)
    with pytest.warns(RuntimeWarning, match="not candidates"):
        out = prob(X)
    assert out.shape == (4, 1)


def test_po_negate() -> None:
    prob_pos = PO128(noise_std=None, negate=False)
    prob_neg = PO128(noise_std=None, negate=True)

    X = prob_pos.candidates()[:2]
    out_pos = prob_pos(X, noise=False)
    out_neg = prob_neg(X, noise=False)
    torch.testing.assert_close(out_pos, -out_neg)


def test_po_no_dim_raises() -> None:
    with pytest.raises(TypeError):
        PO()


def test_po_candidates() -> None:
    prob = PO128(noise_std=None)
    X_cand = prob.candidates()
    assert X_cand.shape == (len(prob.obj_func.ys), 128)
    assert X_cand.dtype == prob.bounds.dtype

    # every candidate evaluates to its own table row
    out = prob(X_cand[:8], noise=False)
    expected = prob.obj_func.ys[:8].unsqueeze(-1).to(dtype=out.dtype)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("dtype", [torch.double, torch.float32])
def test_po_candidates_do_not_warn(dtype: torch.dtype) -> None:
    prob = PO768(noise_std=None).to(dtype=dtype)
    X = prob.candidates(dtype=dtype)[:64]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prob(X, noise=False)
