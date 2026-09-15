"""
ml/train.py
===========
Trains the SurrogateCNN on ml/data/train_pairs.npz (build it first with
`python ml/dataset.py`). CPU-only, small model + small dataset -- a few
seconds per epoch.

Usage:
    python ml/train.py [--epochs 60] [--lr 1e-3] [--batch-size 32]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.model import Normalizer, SurrogateCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "surrogate.pt")

torch.manual_seed(0)


def load_train_data():
    d = np.load(os.path.join(DATA_DIR, "train_pairs.npz"))
    return d["x"], d["y"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args()

    x, y = load_train_data()
    n = x.shape[0]
    n_val = max(1, int(n * args.val_frac))
    perm = np.random.default_rng(0).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    normalizer = Normalizer.fit(x[train_idx], y[train_idx])
    x_norm = normalizer.normalize_x(x)
    residual_norm = (y - x) / normalizer.y_scale.reshape(1, -1, 1, 1)

    train_ds = TensorDataset(torch.from_numpy(x_norm[train_idx]),
                              torch.from_numpy(residual_norm[train_idx]))
    val_x = torch.from_numpy(x_norm[val_idx])
    val_y = torch.from_numpy(residual_norm[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = SurrogateCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    print(f"Training on {len(train_idx)} pairs, validating on {len(val_idx)} "
          f"(model params: {sum(p.numel() for p in model.parameters()):,})")

    history = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= len(train_idx)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()

        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}")

    elapsed = time.perf_counter() - t0
    print(f"Trained {args.epochs} epochs in {elapsed:.1f}s")

    torch.save({
        "model_state": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "history": history,
        "args": vars(args),
    }, CKPT_PATH)
    print(f"Saved checkpoint to {CKPT_PATH}")

    with open(os.path.join(os.path.dirname(__file__), "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
