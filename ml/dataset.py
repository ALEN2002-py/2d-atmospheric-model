"""
ml/dataset.py
=============
Builds a training set of (state_t, state_{t+K}) field pairs for the
coarse-step neural surrogate (see ml/model.py), by running the REAL RK4
solver in src/ -- same G&R (2008) Case 2 cosine-bell bubble and domain
used throughout the rest of this project (experiments/gr_case2_
benchmark.py, api/runner.py), just varying the bubble's own amplitude
across several full simulations to get a diverse set of (state, later
state) transitions for the network to learn the K-step operator from.

This is a parametric sweep over bubble_amp, NOT a general-purpose PDE
dataset -- the surrogate trained on it is scoped accordingly (see the
"Scope" note in ml/README section of the main README).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grid import Grid
from integrators import step

DOMAIN = {"Lx": 1000.0, "Lz": 1000.0, "x_c": 500.0, "z_c": 350.0}
DX = 20.0
BUBBLE_R = 250.0
DT = 0.01 * (DX / 10.0)   # same auto-dt formula used throughout the project
T_END = 40.0
N_STEPS = round(T_END / DT)
K = 50                     # surrogate jump: K solver steps = K*DT seconds
ANCHOR_STRIDE = 20         # spacing between sampled (t, t+K) pairs, in steps

TRAIN_AMPS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TEST_AMPS = [0.35, 0.65]   # held out entirely -- interpolation generalisation test

FIELD_ORDER = ["u", "w", "theta", "pi"]


def make_initial_state(grid: Grid, bubble_amp: float) -> dict:
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - DOMAIN["x_c"]) ** 2 + (grid.z_2d - DOMAIN["z_c"]) ** 2)
    state["theta"] = np.where(
        r <= BUBBLE_R,
        0.5 * bubble_amp * (1.0 + np.cos(np.pi * r / BUBBLE_R)),
        0.0,
    )
    return state


def state_to_array(state: dict) -> np.ndarray:
    """{u,w,theta,pi} (each nz,nx) -> (4, nz, nx) float32 array."""
    return np.stack([state[k] for k in FIELD_ORDER], axis=0).astype(np.float32)


def run_trajectory(bubble_amp: float, verbose: bool = True) -> np.ndarray:
    """Runs RK4 for N_STEPS and returns the full trajectory as (N_STEPS+1, 4, nz, nx)."""
    grid = Grid({"Lx": DOMAIN["Lx"], "Lz": DOMAIN["Lz"], "dx": DX, "dz": DX})
    state = make_initial_state(grid, bubble_amp)

    traj = np.empty((N_STEPS + 1, 4, grid.nz, grid.nx), dtype=np.float32)
    traj[0] = state_to_array(state)

    t0 = time.perf_counter()
    for n in range(N_STEPS):
        state, _, _ = step(state, grid, DT, scheme="RK4")
        traj[n + 1] = state_to_array(state)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  bubble_amp={bubble_amp:.2f}: {N_STEPS} RK4 steps in {elapsed:.1f}s, "
              f"theta_max(t_end)={traj[-1, 2].max():.3f}K")
    return traj


def pairs_from_trajectory(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subsamples (state_t, state_{t+K}) pairs from one trajectory."""
    anchors = list(range(0, traj.shape[0] - K, ANCHOR_STRIDE))
    x = np.stack([traj[i] for i in anchors])
    y = np.stack([traj[i + K] for i in anchors])
    return x, y


def build_dataset(amps: list[float]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for amp in amps:
        traj = run_trajectory(amp)
        x, y = pairs_from_trajectory(traj)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating TRAIN set: amps={TRAIN_AMPS}  "
          f"(dx={DX}m, dt={DT}s, {N_STEPS} steps/{T_END}s, K={K} steps/{K*DT:.1f}s jump)")
    x_train, y_train = build_dataset(TRAIN_AMPS)
    print(f"  -> {x_train.shape[0]} training pairs, shape {x_train.shape[1:]}")
    np.savez_compressed(os.path.join(out_dir, "train_pairs.npz"), x=x_train, y=y_train)

    print(f"\nGenerating TEST trajectories (held out amplitudes): amps={TEST_AMPS}")
    test_trajs = {amp: run_trajectory(amp) for amp in TEST_AMPS}
    np.savez_compressed(
        os.path.join(out_dir, "test_trajectories.npz"),
        **{f"amp_{amp}": traj for amp, traj in test_trajs.items()},
        dt=DT, k=K, n_steps=N_STEPS,
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
