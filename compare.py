import csv
import time
from pathlib import Path

import numpy as np
import torch

import model as pt_mod
from train import evaluate
from train_keras import prepare_arrays, run_keras_experiment


def _get_pytorch_model_cls():
    return next(
        obj for obj in vars(pt_mod).values()
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module)
    )


def _train_pytorch_seed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 2026,
    epochs: int = 30,
    lr: float = 1e-2,
    hidden: int = 16,
    batch_size: int = 64,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=256,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=256,
        shuffle=False,
    )

    net = _get_pytorch_model_cls()(n_features=x_train.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 1
    best_weights = {k: v.detach().clone() for k, v in net.state_dict().items()}

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        n_samples = len(train_loader.dataset)
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        train_loss = total_loss / n_samples
        val_loss, val_mae = evaluate(net, val_loader, loss_fn)
        print(
            f"[PyTorch seed={seed}] Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mae={val_mae:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            best_weights = {k: v.detach().clone() for k, v in net.state_dict().items()}

    elapsed = time.perf_counter() - start
    net.load_state_dict(best_weights)
    _, test_mae = evaluate(net, test_loader, loss_fn)

    return {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": round(float(best_val_mae), 6),
        "test_mae": round(float(test_mae), 6),
        "seconds": round(float(elapsed), 4),
    }


def _train_keras_seed(
    data,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 2026,
    epochs: int = 30,
    lr: float = 1e-2,
    hidden: int = 16,
    batch_size: int = 64,
):
    start = time.perf_counter()
    keras_model, hist = run_keras_experiment(
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        seed=seed,
        plot_path=f"reports/keras_curve_seed_{seed}.png",
        checkpoint_path=f"reports/keras_best_seed_{seed}.keras",
        data=data,
    )
    elapsed = time.perf_counter() - start

    val_losses = hist["val_loss"]
    best_idx = int(np.argmin(val_losses))
    best_epoch = best_idx + 1
    best_val_mae = float(hist["val_mae"][best_idx])

    test_preds = np.asarray(keras_model(x_test, training=False)).reshape(-1)
    test_mae = float(np.mean(np.abs(test_preds - y_test.reshape(-1))))

    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_mae": round(best_val_mae, 6),
        "test_mae": round(test_mae, 6),
        "seconds": round(float(elapsed), 4),
    }


def main(
    seeds: tuple[int, ...] = (2026, 42, 100),
    epochs: int = 30,
    lr: float = 1e-2,
    hidden: int = 16,
    batch_size: int = 64,
) -> None:
    rows = []
    for seed in seeds:
        arrays = prepare_arrays(seed=seed)
        _, _, _, _, x_test, y_test = arrays
        rows.append(
            _train_pytorch_seed(
                *arrays,
                seed=seed, epochs=epochs, lr=lr, hidden=hidden, batch_size=batch_size,
            )
        )
        rows.append(
            _train_keras_seed(
                arrays, x_test, y_test,
                seed=seed, epochs=epochs, lr=lr, hidden=hidden, batch_size=batch_size,
            )
        )

    out_csv = Path("reports/comparison.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["framework", "seed", "best_epoch", "validation_mae", "test_mae", "seconds"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_csv} with {len(rows)} rows.")
    print(f"{'Framework':<10} | {'Seed':<6} | {'BestEp':<6} | {'Val MAE':<10} | {'Test MAE':<10} | {'Seconds':<8}")
    print("-" * 66)
    for r in rows:
        print(
            f"{r['framework']:<10} | {r['seed']:<6} | {r['best_epoch']:<6} | "
            f"{r['validation_mae']:<10.4f} | {r['test_mae']:<10.4f} | {r['seconds']:<8.2f}"
        )


if __name__ == "__main__":
    main()
