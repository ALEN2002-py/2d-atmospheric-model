"""Fast smoke tests for the ML surrogate (ml/). Uses tiny synthetic data,
not the real dataset -- ml/dataset.py's full generation + ml/train.py's
real training run take minutes, which belongs in a manual/dedicated run
(see README's ML Surrogate section for those actual measured numbers),
not in a CI smoke test. This file only checks the code paths don't break:
shapes, the normalizer round-trip, and that one training step doesn't NaN.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

from ml.model import Normalizer, SurrogateCNN, predict_next

H, W = 8, 8


def _synthetic_pairs(n=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4, H, W)).astype("float32")
    y = x + 0.01 * rng.normal(size=(n, 4, H, W)).astype("float32")
    return x, y


def test_normalizer_round_trip():
    x, y = _synthetic_pairs()
    norm = Normalizer.fit(x, y)

    x_norm = norm.normalize_x(x)
    assert x_norm.shape == x.shape
    # normalize_x is a pure per-channel rescale -- inverting it must recover x exactly
    recovered = x_norm * norm.x_scale.reshape(1, -1, 1, 1)
    assert np.allclose(recovered, x, atol=1e-5)


def test_model_forward_shape():
    x, _ = _synthetic_pairs(n=4)
    model = SurrogateCNN()
    out = model(torch.from_numpy(x))
    assert out.shape == x.shape


def test_predict_next_shape_and_finite():
    x, y = _synthetic_pairs()
    norm = Normalizer.fit(x, y)
    model = SurrogateCNN()
    pred = predict_next(model, norm, x[0])
    assert pred.shape == (4, H, W)
    assert np.all(np.isfinite(pred))


def test_one_training_step_does_not_nan():
    x, y = _synthetic_pairs(n=16)
    norm = Normalizer.fit(x, y)
    x_norm = norm.normalize_x(x)
    residual_norm = (y - x) / norm.y_scale.reshape(1, -1, 1, 1)

    model = SurrogateCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    xb, yb = torch.from_numpy(x_norm), torch.from_numpy(residual_norm)
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    opt.step()

    assert np.isfinite(loss.item())
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))
