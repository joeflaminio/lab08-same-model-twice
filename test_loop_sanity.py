from torch import nn, optim

from model import BikeNet
from prepare import Splits, make_loaders, make_splits


def sanity_check():
    # 1. Get one single batch from the training loader
    splits = Splits(*make_splits(2026))
    train_loader, _, _ = make_loaders(splits, batch_size=64)
    xb, yb = next(iter(train_loader))

    # 2. Initialize model, loss function, and optimizer (lr=5e-2 for faster memoization)
    model = BikeNet(splits.n_features, hidden=16)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-2)

    initial_loss = None
    final_loss = None

    # 3. Train on this exact same batch for 300 steps
    model.train()
    for step in range(1, 301):
        optimizer.zero_grad()  # Clear gradients
        preds = model(xb)  # Forward pass
        loss = loss_fn(preds, yb)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        if step == 1:
            initial_loss = loss.item()
        if step == 300:
            final_loss = loss.item()

    print(f"Initial Loss (Step 1): {initial_loss:.4f}")
    print(f"Final Loss (Step 300): {final_loss:.4f}")

    # Assert that the loss dropped significantly close to zero
    assert final_loss < initial_loss * 0.05, "Loss did not drop sufficiently!"
    print("Sanity check passed: The model successfully memorized one small batch!")


if __name__ == "__main__":
    sanity_check()
