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


def test_dm_optima_match_pinned_emulators() -> None:
    """Pin the recorded optima to the emulator revision they were measured on.

    ``_optimal_value``/``_optimizers`` are only valid for the weights named by
    ``hf_revision``, and nothing else in the suite would catch a re-pin that
    silently invalidates them.
    """
    for cls in (DMCurriculum, DMCurriculumHet):
        prob = cls(noise_std=None, negate=False)
        X = torch.tensor([prob._optimizers[0]], dtype=torch.double)
        value = prob(X, noise=False).item()
        assert value == pytest.approx(prob._optimal_value, abs=1e-4), (
            f"{cls.__name__}: emulator gives {value:.6f} at the recorded optimizer, "
            f"but _optimal_value is {prob._optimal_value}"
        )


def test_dm_hetero_noise_shape_and_scale() -> None:
    """Noise is the std of the 3-benchmark average, so it sits near the per-benchmark stds."""
    prob = DMCurriculumHet(negate=False)
    X = _simplex_X(prob)

    noise = prob.evaluate_noise(X)
    assert noise.shape == (16, 1)
    assert (noise > 0).all()
    # averaging 3 near-uncorrelated benchmarks shrinks the std below any single one
    assert (noise < 0.0274).all()
