import copy
import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from model import BikeNet
from prepare import Splits, make_loaders, make_splits


def train_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        total_mae += torch.abs(preds - yb).sum().item()
    n = len(dataloader.dataset)
    return total_loss / n, total_mae / n


def evaluate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    with torch.no_grad():
        for xb, yb in dataloader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * len(xb)
            total_mae += torch.abs(preds - yb).sum().item()
    n = len(dataloader.dataset)
    return total_loss / n, total_mae / n


def compute_baseline(splits: Splits):
    """Computes mean baseline MSE on the test set."""
    y_train = torch.as_tensor(splits.train.y, dtype=torch.float32)
    y_test = torch.as_tensor(splits.test.y, dtype=torch.float32)

    train_mean = y_train.mean()
    baseline_preds = torch.full_like(y_test, train_mean)
    baseline_mse = nn.MSELoss()(baseline_preds, y_test).item()
    return baseline_mse


def run_experiment(epochs: int = 30, lr: float = 1e-2):
    os.makedirs("reports", exist_ok=True)

    splits = Splits(*make_splits(2026))
    train_loader, val_loader, test_loader = make_loaders(splits, batch_size=64)

    model = BikeNet(splits.n_features, hidden=16)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_maes = [], [], []
    best_val_loss = float("inf")
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())

    print(f"=== Starting Training ({epochs} Epochs) ===")
    for epoch in range(1, epochs + 1):
        train_loss, _ = train_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = evaluate(model, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Train MSE: {train_loss:.4f} | "
                f"Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
            )

    # Restore best weights before final test evaluation
    model.load_state_dict(best_weights)
    test_loss, test_mae = evaluate(model, test_loader, loss_fn)
    baseline_mse = compute_baseline(splits)

    print("\n=== Final Results ===")
    print(f"Baseline (Mean Predictor) Test MSE : {baseline_mse:.4f}")
    print(f"Neural Network Test MSE: {test_loss:.4f}")
    print(f"Best Epoch (by Val Loss): {best_epoch}")

    # Plot training curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss (MSE)")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss (MSE)")
    plt.axvline(
        best_epoch,
        color="green",
        linestyle="--",
        label=f"Best Epoch ({best_epoch})",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Lab 07: BikeNet Training and Validation Curves")
    plt.legend()
    plt.grid(True)

    plot_path = "reports/training_curve.png"
    plt.savefig(plot_path)
    print(f"Training curve plot successfully saved to {plot_path}")

    return model

if __name__ == "__main__":
    run_experiment(epochs=30, lr=1e-2)
