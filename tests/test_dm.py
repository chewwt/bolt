from typing import Union

import pytest
import torch

from bolt import DMCurriculum, DMCurriculumHet, DMCurriculumMO


def _simplex_X(
    prob: Union[DMCurriculum, DMCurriculumHet, DMCurriculumMO],
) -> torch.Tensor:
    X = torch.rand((16, prob.dim))
    X[:, 3:] /= X[:, 3:].sum(dim=1)[:, None]
    X[:, :3] /= X[:, :3].sum(dim=1)[:, None]
    return X


def test_dm() -> None:
    prob = DMCurriculum(noise_std=0.001, negate=False)
    out = prob(_simplex_X(prob))
    assert out.shape == (16, 1)


def test_dm_mo() -> None:
    prob = DMCurriculumMO(noise_std=0.001, negate=False)
    out = prob(_simplex_X(prob))
    assert out.shape == (16, 3)

    with pytest.raises(ValueError):
        prob(torch.rand((16, prob.dim)))  # simplex constraint violated


def test_dm_hetero() -> None:
    prob = DMCurriculumHet(negate=False)
    out = prob(_simplex_X(prob))
    assert out.shape == (16, 1)
