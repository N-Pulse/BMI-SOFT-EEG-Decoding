"""Minimal training loop for the CNN on dummy data, to check the pipeline runs end to end.

Run from scripts/CNN:  python train.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from cnn import TCNN

# Dummy dataset size: windows of N_SAMPLES time points over N_CHANNELS electrodes
N_WINDOWS, N_CHANNELS, N_SAMPLES, N_CLASSES = 256, 18, 256, 3
EPOCHS, BATCH_SIZE, LR = 5, 32, 1e-3


def make_dummy_data() -> TensorDataset:
    """Noise windows with a class-dependent offset on one electrode, so the task is learnable."""
    y = torch.randint(0, N_CLASSES, (N_WINDOWS,))       # random class label per window
    x = torch.randn(N_WINDOWS, N_CHANNELS, N_SAMPLES)   # (windows, electrodes, time)
    x[torch.arange(N_WINDOWS), y] += 1.0                # shift electrode `label` of each window (just to put something learnable)
    return TensorDataset(x, y)


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 80/20 train/validation split, batched with a DataLoader (shuffle only the training set)
    train_set, val_set = random_split(make_dummy_data(), [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = TCNN(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()  # applies softmax internally, so the model outputs raw logits

    for epoch in range(1, EPOCHS + 1):
        # --- training pass: update weights after every batch ---
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                # (batch, n_classes)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()            # clear gradients from the previous batch
            loss.backward()                  # backpropagate
            optimizer.step()                 # update weights
            tr_loss += loss.item() * len(y)  # weight by batch size (the last batch may be smaller)
            tr_correct += (logits.argmax(1) == y).sum().item()

        # --- validation pass: no gradients, no weight updates ---
        model.eval()
        va_loss, va_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_loss += loss_fn(logits, y).item() * len(y)
                va_correct += (logits.argmax(1) == y).sum().item()

        # Report per-epoch averages over all samples
        n_tr, n_va = len(train_set), len(val_set)
        print(
            f"epoch {epoch}: train loss {tr_loss / n_tr:.3f} acc {tr_correct / n_tr:.2f} "
            f"| val loss {va_loss / n_va:.3f} acc {va_correct / n_va:.2f}"
        )


if __name__ == "__main__":
    main()
