import pytest
import torch
import torch.nn as nn

from model import BikeNet
from prepare import Splits, make_loaders, make_splits
from train import evaluate, train_epoch


def test_data_splits_and_loaders():
    splits = Splits(*make_splits(2026))
    assert splits.n_features > 0, "Feature count should be greater than zero"

    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=32)
    xb, yb = next(iter(train_loader))
    assert xb.ndim == 2, "Batch features should be 2D"
    assert yb.ndim == 1 or yb.ndim == 2, "Batch targets should be 1D or 2D"


def test_bike_net_forward():
    splits = Splits(*make_splits(2026))
    model = BikeNet(splits.n_features, hidden=16)
    dummy_input = torch.randn(4, splits.n_features)
    preds = model(dummy_input)
    assert preds.shape[0] == 4, "Batch size mismatch"


def test_model_output_shape():
    model = BikeNet(n_features=10, hidden=32)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8,), f"Expected shape (8,), got {out.shape}"


def test_dataset_types():
    splits = Splits(*make_splits(2026))
    train_loader, _, _ = make_loaders(splits, batch_size=16)
    xb, yb = next(iter(train_loader))
    assert xb.dtype == torch.float32, "Features must be float32"
    assert yb.dtype == torch.float32, "Targets must be float32"


@pytest.mark.parametrize("hidden_dim", [8, 16, 32, 64])
def test_bike_net_hidden_dimensions(hidden_dim):
    model = BikeNet(n_features=12, hidden=hidden_dim)
    x = torch.randn(2, 12)
    out = model(x)
    assert out.shape == (2,)


def test_train_epoch_runs():
    splits = Splits(*make_splits(2026))
    train_loader, _, _ = make_loaders(splits, batch_size=64)
    model = BikeNet(splits.n_features, hidden=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    loss, mae = train_epoch(model, train_loader, loss_fn, optimizer)
    assert isinstance(loss, float)
    assert isinstance(mae, float)
    assert loss >= 0.0


def test_evaluate_runs():
    splits = Splits(*make_splits(2026))
    _, val_loader, _ = make_loaders(splits, batch_size=64)
    model = BikeNet(splits.n_features, hidden=8)
    loss_fn = torch.nn.MSELoss()

    loss, mae = evaluate(model, val_loader, loss_fn)
    assert isinstance(loss, float)
    assert isinstance(mae, float)
    assert loss >= 0.0