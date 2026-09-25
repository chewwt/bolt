import pytest
import torch

from bolt import PCO16, PCO32, PCO64
from bolt.problems.parallelism_config import OOM_MARGIN_PAD, PCO

# (class, table size, feasible rows)
PROBLEMS = [(PCO16, 919, 632), (PCO32, 379, 339), (PCO64, 780, 560)]


@pytest.fixture(scope="module", params=PROBLEMS, ids=lambda p: p[0].__name__)
def prob_info(request) -> tuple[PCO, int, int]:
    cls, n_rows, n_feasible = request.param
    return cls(), n_rows, n_feasible


def test_pco_table(prob_info: tuple[PCO, int, int]) -> None:
    prob, n_rows, n_feasible = prob_info
    X = prob.candidates()
    assert X.shape == (n_rows, prob.dim)
    assert X.dtype == torch.double
    assert int(prob.feasible.sum()) == n_feasible

    # every row within bounds, zero_stage encoded as an ordinal
    assert (X >= prob.bounds[0]).all() and (X <= prob.bounds[1]).all()
    zero_col = prob.input_cols.index("zero_stage")
    assert set(X[:, zero_col].tolist()) <= {0.0, 1.0, 2.0}


def test_pco_optimum(prob_info: tuple[PCO, int, int]) -> None:
    prob, _, _ = prob_info
    X_opt = torch.tensor(prob._optimizers, dtype=torch.double)
    y_opt = prob(X_opt, noise=False)
    torch.testing.assert_close(
        y_opt, torch.tensor([[prob._optimal_value]], dtype=torch.double)
    )
    assert prob.is_feasible(X_opt, noise=False).all()
    # the optimum is the best feasible row
    torch.testing.assert_close(
        prob.obj_func.ys[prob.feasible].max().to(torch.double),
        torch.tensor(prob._optimal_value, dtype=torch.double),
    )


def test_pco_imputation(prob_info: tuple[PCO, int, int]) -> None:
    prob, n_rows, _ = prob_info
    X = prob.candidates()
    feasible = prob.feasible  # ground truth, row-aligned with candidates()

    y = prob(X, noise=False)
    y_table = prob.obj_func.ys.to(dtype=y.dtype).unsqueeze(-1)
    assert y.shape == (n_rows, 1)

    # OOM rows record no throughput; the objective fills them in
    assert (y_table.isnan().squeeze(-1) == ~feasible).all()
    assert not y.isnan().any()
    # imputation only touches OOM rows
    torch.testing.assert_close(y[feasible], y_table[feasible])


def test_pco_slack(prob_info: tuple[PCO, int, int]) -> None:
    prob, n_rows, _ = prob_info
    X = prob.candidates()
    feasible = prob.feasible  # ground truth, row-aligned with candidates()

    slack = prob.evaluate_slack(X, noise=False)
    assert slack.shape == (n_rows, 1)
    assert ((slack.squeeze(-1) > 0) == feasible).all()
    assert (prob.is_feasible(X, noise=False) == feasible).all()
    torch.testing.assert_close(
        slack[~feasible],
        torch.full_like(slack[~feasible], -OOM_MARGIN_PAD),
    )

    mem = prob.evaluate_peak_mem(X).squeeze(-1)
    assert mem[~feasible].isinf().all()
    assert (mem[feasible] <= prob.mem_capacity_gb).all()


def test_pco_snaps_off_table(prob_info: tuple[PCO, int, int]) -> None:
    prob, _, _ = prob_info
    # an integer point in bounds but not in the table: step one row's first
    # coordinate until it leaves the table
    X_cand = prob.candidates()
    X = None
    for row in X_cand:
        for step in (1.0, -1.0):
            x = row.clone()
            x[0] += step
            in_bounds = prob.bounds[0, 0] <= x[0] <= prob.bounds[1, 0]
            if in_bounds and torch.cdist(x[None], X_cand).min() > 0:
                X = x[None]
                break
        if X is not None:
            break
    assert X is not None
    with pytest.warns(RuntimeWarning, match="not in the table"):
        y = prob(X, noise=False)
    nearest = X_cand[torch.cdist(X, X_cand).argmin(dim=1)]
    torch.testing.assert_close(y, prob(nearest, noise=False))


def test_pco_negate() -> None:
    prob_pos = PCO32(negate=False)
    prob_neg = PCO32(negate=True)
    X = prob_pos.candidates()[:4]
    torch.testing.assert_close(prob_pos(X, noise=False), -prob_neg(X, noise=False))
