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
    out = prob(X)
    assert out.shape == (4, 1)


def test_po_negate() -> None:
    prob_pos = PO128(noise_std=None, negate=False)
    prob_neg = PO128(noise_std=None, negate=True)

    X = torch.zeros((2, 128), dtype=torch.double)
    out_pos = prob_pos(X, noise=False)
    out_neg = prob_neg(X, noise=False)
    torch.testing.assert_close(out_pos, -out_neg)


def test_po_no_dim_raises() -> None:
    with pytest.raises(TypeError):
        PO()
