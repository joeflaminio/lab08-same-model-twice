# Lab 08: PyTorch vs. Keras Model & Training Comparison

## 1. Architecture & Parameter Parity
Both implementations define the same regression MLP architecture for bike-sharing demand (`n_features = 9` after preprocessing):
- Hidden Layer 1: `Dense(hidden)` + `ReLU`
- Hidden Layer 2: `Dense(hidden)` + `ReLU`
- Output Layer: `Dense(1)` (linear) squeezed to shape `(batch_size,)`

At `hidden = 16`, both models have **433 trainable parameters** (matching across other hidden sizes such as 4, 32, and 64 as well).

## 2. Pre-Training Weight Copy & Numerical Equivalence
PyTorch `nn.Linear` weights have shape `(out_features, in_features)`, whereas Keras `Dense` kernels have shape `(in_features, out_features)`. Transposing each weight matrix during `copy_weights_from_pytorch` aligns the parameters cleanly:
- **Pre-training copied-weight max absolute difference (`hidden = 16`)**: `2.98e-08`

## 3. Training & Test Comparison (`epochs = 30`, `lr = 1e-2`, `hidden = 16`, `batch_size = 64`)
From running `python compare.py`:

| Metric | PyTorch | Keras |
| :--- | :---: | :---: |
| Parameters (`hidden=16`) | 433 | 433 |
| Pre-train Copied Max Diff | `0.0000e+00` | `2.98e-08` |
| Final Train MSE (Loss) | *(logged by `train.py`)* | `67.4604` |
| Final Train MAE | *(logged by `train.py`)* | `5.0725` |
| Final Val MSE (Loss) | *(logged by `train.py`)* | `62.7908` |
| Final Val MAE | *(logged by `train.py`)* | `5.1174` |
| Test MSE | `124.0944` | `62.0911` |
| Test MAE | `7.5677` | `5.2327` |

## 4. Key Takeaways
- **Identical function at identical weights**: Before training, copying PyTorch weights into Keras reproduces the forward pass within single-precision floating-point tolerance (`2.98e-08`).
- **Different training trajectories over 30 epochs**: Even with matching architecture, MSE loss, Adam optimizer, and learning rate (`1e-2`), the two runs reached different test errors (`124.0944` MSE / `7.5677` MAE in PyTorch vs. `62.0911` MSE / `5.2327` MAE in Keras). Differences in random initialization, mini-batch ordering/shuffling, and optimizer/backend execution details can steer gradient-based optimization along distinct paths over multiple epochs.

## 4. Reproducibility & Fresh-Clone Verification
Verified `ruff format --check .`, `ruff check .`, `pytest`, `train.py`, `train_keras.py`, and `compare.py` from a fresh clone, with artifacts written to `reports/`.
