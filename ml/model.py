"""
ml/model.py
===========
A small residual CNN that learns the K-step solver operator:
    state_{t+K} ~= state_t + f_theta(state_t)

Predicting the residual (change over K steps) rather than the absolute
next state is standard practice for PDE surrogates -- at this dataset's
K*dt=1.0s jump the fields change by a small fraction of their own scale
per jump (see ml/dataset.py), so the residual is much easier to regress
than the raw field, and the network only has to learn "how much do things
change", not memorise the coordinate-shaped bubble itself.

Purely local, translation-equivariant convolutions are enough for this
scope: at the velocities produced in this dataset's regime (bubble_amp
<= 0.8K, t <= 40s), advection moves the bubble by a small fraction of one
grid cell per K-step jump, well within a handful of stacked 3x3 conv
receptive fields.
"""

import numpy as np
import torch
import torch.nn as nn

N_CHANNELS = 4  # u, w, theta, pi


class SurrogateCNN(nn.Module):
    def __init__(self, channels: int = N_CHANNELS, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """x_norm: normalised state (B, 4, nz, nx) -> normalised residual (B, 4, nz, nx)."""
        return self.net(x_norm)


class Normalizer:
    """Per-channel scale computed from the training set; same object is
    saved with the checkpoint and reused at eval/inference time so the
    two are never accidentally computed on different statistics."""

    def __init__(self, x_scale, y_scale):
        self.x_scale = x_scale  # (4,) float32 -- std of each input channel
        self.y_scale = y_scale  # (4,) float32 -- std of each residual channel

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "Normalizer":
        x_scale = np.maximum(x.std(axis=(0, 2, 3)), 1e-8).astype("float32")
        residual = y - x
        y_scale = np.maximum(residual.std(axis=(0, 2, 3)), 1e-8).astype("float32")
        return cls(x_scale, y_scale)

    def normalize_x(self, x):
        return x / self.x_scale.reshape(1, -1, 1, 1)

    def denormalize_residual(self, r_norm):
        return r_norm * self.y_scale.reshape(1, -1, 1, 1)

    def to_dict(self):
        return {"x_scale": self.x_scale, "y_scale": self.y_scale}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x_scale"], d["y_scale"])


def predict_next(model: SurrogateCNN, normalizer: Normalizer, state: np.ndarray) -> np.ndarray:
    """state (4, nz, nx) physical units -> state_{t+K} (4, nz, nx) physical units.

    Numpy in, numpy out -- this is the function ml/evaluate.py's rollout
    loop calls once per K-step jump, so it can chain directly against the
    same (4, nz, nx) array shape the real solver's states are stacked into.
    """
    model.eval()
    batched = state[None, ...]  # (1, 4, nz, nx)
    x_norm = torch.from_numpy(normalizer.normalize_x(batched))
    with torch.no_grad():
        residual_norm = model(x_norm).numpy()
    residual = normalizer.denormalize_residual(residual_norm)
    return (batched + residual)[0]
