import torch
import torch.nn.functional as F

from bolt import HPO, HPOMultiFidelityModel, HPOMultiFidelityToken
from bolt._utils import unstandardize_y


def test_hpo() -> None:
    prob = HPO(noise_std=0.001, negate=False)

    X = torch.rand((16, prob.dim))
    bounds = torch.Tensor(prob._bounds)
    X = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    for i in prob.discrete_inds:
        X[:, i] = X[:, i].int()

    for i in prob.categorical_inds:
        X[:, i] = X[:, i].round().clamp(0, 3).long()

    out = prob(X)
    assert out.shape == (16, 1)


def test_hpo_fd_step() -> None:
    prob = HPOMultiFidelityToken(noise_std=0.001, negate=False)

    X = torch.rand((16, prob.dim))
    bounds = torch.Tensor(prob._bounds)
    X = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    for i in prob.discrete_inds:
        X[:, i] = X[:, i].int()

    for i in prob.categorical_inds:
        X[:, i] = X[:, i].round().clamp(0, 3).long()

    out = prob(X)
    assert out.shape == (16, 1)

    cost = prob.cost(X)
    assert cost.shape == (16,)


def test_hpo_fd_step_monotonic() -> None:
    prob = HPOMultiFidelityToken(noise_std=None, negate=False)

    # Fixed hyperparameters (columns 0-6), varying fidelity (column 7)
    fidelities = torch.linspace(0.05, 1.0, 10)
    base = torch.tensor([[0.5, 3, 3, 3, 0.2, 15, 1]]).expand(len(fidelities), -1)
    X = torch.cat([base, fidelities.unsqueeze(1)], dim=1)

    out = prob(X, noise=False)  # shape (10, 1)
    values = out.squeeze(1)
    assert (values[1:] >= values[:-1]).all(), (
        "HPOMultiFidelityToken output must be monotonically non-decreasing with fidelity"
    )


def test_hpo_fd_model() -> None:
    prob = HPOMultiFidelityModel(noise_std=0.001, negate=False)

    X = torch.rand((2, prob.dim - 1))
    bounds = torch.Tensor(prob._bounds)

    # high fidelity
    X_hf = torch.concat((X.clone().detach(), torch.ones(X.shape[0], 1)), dim=1)
    X_hf = X_hf * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    for i in prob.discrete_inds:
        X_hf[:, i] = X_hf[:, i].int()

    for i in prob.categorical_inds:
        X_hf[:, i] = X_hf[:, i].round().clamp(0, 3).long()

    out = prob(X_hf, noise=False)

    for i in prob.discrete_inds:
        i_min, i_max = prob._bounds[i]
        X_hf[:, i] = (X_hf[:, i].float() - i_min) / (i_max - i_min)

    one_hot = F.one_hot(X_hf[:, 6].long(), num_classes=4).float()
    X_hf_enc = torch.cat(
        [X_hf[:, :6], one_hot, torch.ones(len(X_hf), 1)], dim=1
    )  # ones for tokens
    out_obj0 = prob.high_fidelity_obj_function.evaluate_true(X_hf_enc)
    out_obj0 = unstandardize_y(out_obj0, prob.y_mean_high_fid, prob.y_std_high_fid)

    assert out.shape == (2, 1)
    torch.testing.assert_close(out, out_obj0)

    # low fidelity
    X_lf = torch.concat((X.clone().detach(), torch.zeros(X.shape[0], 1)), dim=1)
    X_lf = X_lf * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    for i in prob.discrete_inds:
        X_lf[:, i] = X_lf[:, i].int()

    for i in prob.categorical_inds:
        X_lf[:, i] = X_lf[:, i].round().clamp(0, 3).long()

    out = prob(X_lf, noise=False)

    for i in prob.discrete_inds:
        i_min, i_max = prob._bounds[i]
        X_lf[:, i] = (X_lf[:, i] - i_min) / (i_max - i_min)

    one_hot = F.one_hot(X_lf[:, 6].long(), num_classes=4).float()
    X_lf_enc = torch.cat([X_lf[:, :6], one_hot, torch.ones(len(X_lf), 1)], dim=1)
    out_obj1 = prob.low_fidelity_obj_function.evaluate_true(X_lf_enc)
    out_obj1 = unstandardize_y(out_obj1, prob.y_mean_low_fid, prob.y_std_low_fid)

    assert out.shape == (2, 1)
    torch.testing.assert_close(out, out_obj1)

    cost = prob.cost(X)
    assert cost.shape == (2,)


def test_hpo_fd_model_mixed_fidelity_ordering() -> None:
    """Verify that outputs from a mixed-fidelity batch are in the correct order."""
    prob = HPOMultiFidelityModel(noise_std=None, negate=False)

    bounds = torch.Tensor(prob._bounds)

    def make_X(n: int, fidelity: int) -> torch.Tensor:
        X = torch.rand((n, prob.dim - 1))
        X_full = torch.cat([X, torch.full((n, 1), fidelity)], dim=1)
        X_full = X_full * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

        for i in prob.discrete_inds:
            X_full[:, i] = X_full[:, i].int()

        for i in prob.categorical_inds:
            X_full[:, i] = X_full[:, i].round().clamp(0, 3).long()

        return X_full

    torch.manual_seed(0)
    X_hf = make_X(2, fidelity=0)  # high fidelity (fid=0)
    X_lf = make_X(2, fidelity=1)  # low fidelity (fid=1)

    # interleaved: [hf[0], lf[0], hf[1], lf[1]]
    X_mixed = torch.stack([X_hf[0], X_lf[0], X_hf[1], X_lf[1]])

    out_hf = prob(X_hf, noise=False)
    out_lf = prob(X_lf, noise=False)
    out_mixed = prob(X_mixed, noise=False)

    torch.testing.assert_close(out_mixed[0], out_hf[0])
    torch.testing.assert_close(out_mixed[1], out_lf[0])
    torch.testing.assert_close(out_mixed[2], out_hf[1])
    torch.testing.assert_close(out_mixed[3], out_lf[1])
