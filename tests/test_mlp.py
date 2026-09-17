import pytest
import torch

from bolt.functions.mlp import FeatureNet


def test_output_shape():
    model = FeatureNet(input_dim=6, hidden_dim=256, output_dim=3)
    out = model(torch.randn(16, 6))
    assert out.shape == (16, 3)


def test_wrong_input_dim_raises():
    model = FeatureNet(input_dim=6, hidden_dim=256, output_dim=3)
    with pytest.raises(RuntimeError):
        model.mlp(torch.randn(16, 5))


def test_n_layers_default_matches_two_blocks():
    """The default reproduces the architecture used before depth was configurable."""
    default = FeatureNet(input_dim=6, hidden_dim=32, output_dim=1)
    explicit = FeatureNet(input_dim=6, hidden_dim=32, output_dim=1, n_layers=2)
    assert default.state_dict().keys() == explicit.state_dict().keys()


def test_n_layers_changes_depth():
    model = FeatureNet(input_dim=6, hidden_dim=32, output_dim=3, n_layers=3)
    out = model(torch.randn(16, 6))
    assert out.shape == (16, 3)

    n_linear = sum(1 for m in model.mlp if isinstance(m, torch.nn.Linear))
    assert n_linear == 4  # 3 hidden blocks + output layer


def test_n_layers_below_one_raises():
    with pytest.raises(ValueError, match="n_layers must be >= 1"):
        FeatureNet(input_dim=6, hidden_dim=32, output_dim=1, n_layers=0)
