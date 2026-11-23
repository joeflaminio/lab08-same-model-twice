from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from keras_model import build_keras_model
from prepare import make_preprocessor, make_splits


def _to_numpy(val) -> np.ndarray:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return val.to_numpy()
    return np.asarray(val)


def _extract_xy(split):
    if hasattr(split, "tensors"):
        x_t, y_t = split.tensors
        return _to_numpy(x_t), _to_numpy(y_t).astype(np.float32).reshape(-1)
    if isinstance(split, (tuple, list)) and len(split) == 2:
        x_arr, y_arr = split
        return x_arr, _to_numpy(y_arr).astype(np.float32).reshape(-1)
    if isinstance(split, pd.DataFrame):
        target_col = next(
            (col for col in ("cnt", "target", "count", "bikes", "y") if col in split.columns),
            split.columns[-1],
        )
        return split.drop(columns=[target_col]), split[target_col].to_numpy().astype(np.float32).reshape(-1)
    xs, ys = [], []
    for x_item, y_item in split:
        xs.append(_to_numpy(x_item))
        ys.append(_to_numpy(y_item))
    return np.stack(xs), np.stack(ys).astype(np.float32).reshape(-1)


def _get_transformer(preprocessor_obj):
    if hasattr(preprocessor_obj, "fit_transform"):
        return preprocessor_obj
    if hasattr(preprocessor_obj, "preprocessor") and hasattr(preprocessor_obj.preprocessor, "fit_transform"):
        return preprocessor_obj.preprocessor
    raise TypeError("Unsupported preprocessor object returned by make_preprocessor()")


def prepare_arrays(seed: int = 2026):
    train_split, val_split, test_split = make_splits(seed=seed)
    transformer = _get_transformer(make_preprocessor())
    x_train_raw, y_train = _extract_xy(train_split)
    x_val_raw, y_val = _extract_xy(val_split)
    x_test_raw, y_test = _extract_xy(test_split)
    return (
        np.asarray(transformer.fit_transform(x_train_raw), dtype=np.float32),
        y_train,
        np.asarray(transformer.transform(x_val_raw), dtype=np.float32),
        y_val,
        np.asarray(transformer.transform(x_test_raw), dtype=np.float32),
        y_test,
    )


def run_keras_experiment(
    epochs: int = 30,
    lr: float = 1e-2,
    hidden: int = 16,
    batch_size: int = 64,
    seed: int = 2026,
    plot_path: str = "reports/keras_training_curve.png",
    checkpoint_path: str = "reports/keras_best.keras",
    data=None,
):
    keras.utils.set_random_seed(seed)
    x_train, y_train, x_val, y_val, _, _ = prepare_arrays(seed=seed) if data is None else data

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    model = build_keras_model(n_features=x_train.shape[1], hidden=hidden)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )

    hist = history.history
    n_ep = len(hist["loss"])
    for i in range(n_ep):
        print(
            f"Epoch {i + 1:02d}/{epochs:02d} | "
            f"train_loss={hist['loss'][i]:.4f} | val_loss={hist['val_loss'][i]:.4f} | val_mae={hist['val_mae'][i]:.4f}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_ep + 1), hist["loss"], label="Train Loss (MSE)")
    plt.plot(range(1, n_ep + 1), hist["val_loss"], label="Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Lab 08: Keras Training and Validation Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Keras training curve plot successfully saved to {plot_path}")
    return model, hist


if __name__ == "__main__":
    run_keras_experiment(epochs=30, lr=1e-2)
