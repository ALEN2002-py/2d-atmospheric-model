"""
ml/evaluate.py
==============
Honest evaluation of the trained surrogate against the real RK4 solver on
the two held-out bubble amplitudes from ml/dataset.py (never seen during
training). Reports two separate error numbers per test trajectory, since
they answer different questions:

  - "single-step" error: feed the surrogate GROUND-TRUTH state_t each
    time and compare its state_{t+K} prediction to the true state_{t+K}.
    Isolates the model's own one-jump accuracy.
  - "rollout" error: feed the surrogate its OWN previous prediction back
    in, chained across the whole trajectory. This is the harder, more
    realistic test (compounding error), and the one that actually
    matters if the surrogate were used to replace repeated solver calls.

Also measures wall-clock time for the real solver vs the surrogate over
the same span, run on THIS machine, right now -- not looked up or
assumed. See README's ML Surrogate section for the actual numbers this
script produced and their scope.

Usage:
    python ml/evaluate.py
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid import Grid
from integrators import step
from ml.dataset import DOMAIN, DT, DX, K, make_initial_state
from ml.model import Normalizer, SurrogateCNN, predict_next
from ml.train import CKPT_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIELD_NAMES = ["u", "w", "theta", "pi"]


def load_model():
    ckpt = torch.load(CKPT_PATH, weights_only=False)
    model = SurrogateCNN()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    normalizer = Normalizer.from_dict(ckpt["normalizer"])
    return model, normalizer


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """Combined error over all 4 stacked channels. Kept for reference, but
    NOT the headline number: since u/w/theta/pi differ by orders of
    magnitude in scale, this single number is dominated by whichever
    channel the joint L2 norm happens to weight most heavily, not by
    which field the surrogate actually gets wrong -- see per_channel_l2."""
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-8))


def per_channel_l2(pred: np.ndarray, true: np.ndarray) -> dict:
    return {name: relative_l2(pred[i], true[i]) for i, name in enumerate(FIELD_NAMES)}


def evaluate_trajectory(model, normalizer, traj: np.ndarray, amp: float):
    n_jumps = (traj.shape[0] - 1) // K
    checkpoints = [j * K for j in range(n_jumps + 1)]

    single_step_err, rollout_err = [], []
    single_step_per_channel, rollout_per_channel = [], []
    rollout_state = traj[0].copy()
    for j in range(1, n_jumps + 1):
        true_prev, true_now = traj[checkpoints[j - 1]], traj[checkpoints[j]]

        single_pred = predict_next(model, normalizer, true_prev)
        single_step_err.append(relative_l2(single_pred, true_now))
        single_step_per_channel.append(per_channel_l2(single_pred, true_now))

        rollout_state = predict_next(model, normalizer, rollout_state)
        rollout_err.append(relative_l2(rollout_state, true_now))
        rollout_per_channel.append(per_channel_l2(rollout_state, true_now))

    return {
        "amp": amp,
        "t_checkpoints": [c * DT for c in checkpoints[1:]],
        "single_step_relative_l2": single_step_err,
        "rollout_relative_l2": rollout_err,
        "single_step_per_channel": single_step_per_channel,
        "rollout_per_channel": rollout_per_channel,
        "final_rollout_state": rollout_state,
        "final_true_state": traj[-1],
    }


def time_real_solver(bubble_amp: float, n_jumps: int) -> float:
    """Wall time for the real RK4 solver to cover n_jumps*K steps, right now."""
    grid = Grid({"Lx": DOMAIN["Lx"], "Lz": DOMAIN["Lz"], "dx": DX, "dz": DX})
    state = make_initial_state(grid, bubble_amp)
    t0 = time.perf_counter()
    for _ in range(n_jumps * K):
        state, _, _ = step(state, grid, DT, scheme="RK4")
    return time.perf_counter() - t0


def time_surrogate(model, normalizer, traj0: np.ndarray, n_jumps: int) -> float:
    state = traj0.copy()
    t0 = time.perf_counter()
    for _ in range(n_jumps):
        state = predict_next(model, normalizer, state)
    return time.perf_counter() - t0


def plot_comparison(results: list[dict], grid: Grid, out_path: str):
    fig, axes = plt.subplots(len(results), 3, figsize=(11, 4 * len(results)))
    if len(results) == 1:
        axes = axes[None, :]
    for row, res in enumerate(results):
        true_theta = res["final_true_state"][2]
        pred_theta = res["final_rollout_state"][2]
        vmax = max(0.05, float(np.max(np.abs(true_theta))))
        levels = np.linspace(0, vmax, 21)

        for col, (field, title) in enumerate([
            (true_theta, f"RK4 (truth), amp={res['amp']}"),
            (pred_theta, f"Surrogate rollout, amp={res['amp']}"),
            (pred_theta - true_theta, "Error (surrogate - truth)"),
        ]):
            ax = axes[row, col]
            if col < 2:
                cf = ax.contourf(grid.x_2d / 1000, grid.z_2d / 1000, field,
                                  levels=levels, cmap="turbo", extend="both")
            else:
                emax = max(1e-4, float(np.max(np.abs(field))))
                cf = ax.contourf(grid.x_2d / 1000, grid.z_2d / 1000, field,
                                  levels=np.linspace(-emax, emax, 21), cmap="RdBu_r")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("x (km)")
            ax.set_ylabel("z (km)")
            fig.colorbar(cf, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    model, normalizer = load_model()
    test_data = np.load(os.path.join(DATA_DIR, "test_trajectories.npz"))
    amps = [float(k.replace("amp_", "")) for k in test_data.files if k.startswith("amp_")]

    results = []
    for amp in amps:
        traj = test_data[f"amp_{amp}"]
        print(f"\nEvaluating amp={amp} ({(traj.shape[0]-1)//K} K-step jumps)...")
        res = evaluate_trajectory(model, normalizer, traj, amp)
        results.append(res)
        print(f"  single-step relative L2 (combined, see note): "
              f"mean={np.mean(res['single_step_relative_l2']):.4f}")
        print(f"  rollout     relative L2 (combined, see note): "
              f"mean={np.mean(res['rollout_relative_l2']):.4f}  final={res['rollout_relative_l2'][-1]:.4f}")
        final_per_channel = res["rollout_per_channel"][-1]
        print("  rollout relative L2 by field, at t={:.0f}s (final): {}".format(
            res["t_checkpoints"][-1],
            "  ".join(f"{k}={v:.3f}" for k, v in final_per_channel.items())))

    n_jumps = (test_data[f"amp_{amps[0]}"].shape[0] - 1) // K
    print(f"\nTiming {n_jumps} K-step jumps ({n_jumps * K} real solver steps, "
          f"{n_jumps * K * DT:.0f}s simulated) on THIS machine, right now:")
    real_time = time_real_solver(amps[0], n_jumps)
    surrogate_time = time_surrogate(model, normalizer, test_data[f"amp_{amps[0]}"][0], n_jumps)
    speedup = real_time / surrogate_time
    print(f"  real RK4 solver : {real_time:.3f}s")
    print(f"  surrogate CNN   : {surrogate_time:.3f}s")
    print(f"  speedup         : {speedup:.1f}x")

    grid = Grid({"Lx": DOMAIN["Lx"], "Lz": DOMAIN["Lz"], "dx": DX, "dz": DX})
    fig_path = os.path.join(os.path.dirname(__file__), "surrogate_comparison.png")
    plot_comparison(results, grid, fig_path)
    print(f"\nSaved comparison figure to {fig_path}")

    summary = {
        "domain": DOMAIN, "dx": DX, "dt": DT, "k_steps": K, "k_seconds": K * DT,
        "n_jumps_timed": n_jumps,
        "real_solver_wall_s": real_time,
        "surrogate_wall_s": surrogate_time,
        "speedup_x": speedup,
        "per_trajectory": [
            {"amp": r["amp"],
             "single_step_relative_l2_combined_mean": float(np.mean(r["single_step_relative_l2"])),
             "rollout_relative_l2_combined_mean": float(np.mean(r["rollout_relative_l2"])),
             "rollout_relative_l2_combined_final": float(r["rollout_relative_l2"][-1]),
             "rollout_relative_l2_by_field_final": r["rollout_per_channel"][-1],
             "single_step_relative_l2_by_field_mean": {
                 name: float(np.mean([c[name] for c in r["single_step_per_channel"]]))
                 for name in FIELD_NAMES
             },
             "rollout_relative_l2_by_field_mean": {
                 name: float(np.mean([c[name] for c in r["rollout_per_channel"]]))
                 for name in FIELD_NAMES
             }}
            for r in results
        ],
    }
    with open(os.path.join(os.path.dirname(__file__), "evaluation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(os.path.dirname(__file__), 'evaluation_results.json')}")


if __name__ == "__main__":
    main()
