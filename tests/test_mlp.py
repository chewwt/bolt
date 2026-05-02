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
