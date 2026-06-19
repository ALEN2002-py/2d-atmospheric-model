"""
experiments/warm_bubble.py
==========================
Warm bubble benchmark — Robert (1993).

Runs all available schemes at matched settings and saves results.
Produces heatmap plots for comparison.

Usage
-----
    python experiments/warm_bubble.py
    python experiments/warm_bubble.py --scheme CTCS --dt 0.01 --n_steps 1000
"""

import argparse
import sys
import os
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from grid        import Grid
from integrators import step, robert_asselin_filter
from results     import save_experiment

PLOT_DIR = "output/figures"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("output/results", exist_ok=True)


def run_warm_bubble(scheme='CTCS', bubble_amp=2.0, dt=0.01,
                    n_steps=1000, save=True):
    """
    Run warm bubble and return snapshots.

    dt=0.01 s gives CFL_1D=0.35, CFL_2D=0.49, safely below the leapfrog
    2-D acoustic stability limit (CFL_2D <= 1).
    dt=0.02 places CTCS at CFL_2D~0.98 (essentially at the boundary);
    with the correct Robert-Asselin filter it will blow up.
    """
    grid      = Grid({"Lx": 1000.0, "Lz": 1000.0,
                      "dx": 10.0,   "dz": 10.0})
    state     = grid.allocate_state()
    state_old = None  # CTCS bootstraps with FTCS on step 0 when None

    # Gaussian warm bubble initial condition
    xc   = grid.Lx / 2.0
    zc   = grid.Lz * 0.4
    r    = 150.0
    r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
    state["theta"] = bubble_amp * np.exp(-r_sq / r**2)

    gamma   = grid.cp / grid.cv
    c_sound = np.sqrt(gamma * grid.Rd * grid.T0)
    cfl     = c_sound * dt / grid.dx
    cfl_2d  = cfl * np.sqrt(2)
    print(f"  [{scheme}] CFL_1D={cfl:.3f}  CFL_2D={cfl_2d:.3f}  "
          f"T={dt*n_steps:.1f}s  A={bubble_amp}K")

    snapshots  = [(0.0, {k: v.copy() for k, v in state.items()})]
    save_every = max(1, n_steps // 5)
    t          = 0.0
    blown_up   = False

    for n in range(n_steps):
        # Save phi^(n-1) BEFORE step() overwrites state_old with phi^n
        state_prev = state_old
        state_new, state_old, _ = step(state, grid, dt,
                                       scheme=scheme,
                                       state_old=state_old)
        if scheme == 'CTCS' and n > 0:
            # Proper Robert-Asselin filter:
            #   phi^n_f = phi^n + alpha*(phi^(n-1) - 2*phi^n + phi^(n+1))
            state_filtered = robert_asselin_filter(state_prev, state,
                                                   state_new, alpha=0.1)
            # For next leapfrog: current=phi^(n+1), old=phi^n_filtered
            state_old = state_filtered
            state = state_new
        else:
            state = state_new

        t += dt

        if np.any(np.isnan(state['u'])) or np.any(np.isinf(state['u'])):
            print(f"  [{scheme}] BLOW-UP at step {n+1} (t={t:.2f}s)")
            blown_up = True
            break

        if (n + 1) % save_every == 0:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

    if not blown_up:
        print(f"  [{scheme}] Complete. "
              f"|w|_max={np.max(np.abs(state['w'])):.3e}  "
              f"|theta|_max={np.max(np.abs(state['theta'])):.3e}")
        if save:
            save_experiment(
                name      = f"warm_bubble_{scheme}_dt{dt}",
                state     = state,
                grid      = grid,
                snapshots = snapshots,
                metadata  = {"scheme": scheme, "dt": dt,
                             "n_steps": n_steps,
                             "bubble_amp": bubble_amp,
                             "experiment": "warm_bubble"}
            )

    return snapshots, grid, blown_up


def plot_bubble_snapshots(snapshots, grid, scheme, variable='theta'):
    """Plot 2D heatmaps at each saved time."""
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4))
    if n == 1:
        axes = [axes]

    var_label = "theta'" if variable == 'theta' else variable
    fig.suptitle(
        f"Warm Bubble -- {var_label}  |  {scheme}",
        fontsize=12, fontweight='bold'
    )

    vals = np.concatenate([s[variable].ravel() for _, s in snapshots])
    vmax = max(np.max(np.abs(vals)), 1e-10)
    # Symmetric diverging colormap so cold anomalies (acoustic waves) are visible.
    # Using vmin=0 would clip negative theta' values and hide real physics.
    vmin = -vmax
    cmap = 'RdBu_r'  # red=warm/positive, blue=cold/negative

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0

    for ax, (t, state) in zip(axes, snapshots):
        im = ax.pcolormesh(x_km, z_km, state[variable],
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto')
        ax.set_title(f"t = {t:.1f} s", fontsize=10)
        ax.set_xlabel("x (km)", fontsize=8)
        ax.set_aspect('equal')
        ax.axhline(0.8, color='dodgerblue', lw=0.7,
                   linestyle='--', alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("z (km)", fontsize=8)
    plt.tight_layout()

    fname = os.path.join(PLOT_DIR, f"warm_bubble_{scheme}_{variable}.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme",     default="CTCS")
    parser.add_argument("--bubble_amp", type=float, default=2.0)
    parser.add_argument("--dt",         type=float, default=0.01)
    parser.add_argument("--n_steps",    type=int,   default=1000)
    args = parser.parse_args()

    snaps, grid, blown = run_warm_bubble(
        scheme=args.scheme,
        bubble_amp=args.bubble_amp,
        dt=args.dt,
        n_steps=args.n_steps,
    )

    if not blown:
        plot_bubble_snapshots(snaps, grid, args.scheme, 'theta')
        plot_bubble_snapshots(snaps, grid, args.scheme, 'w')
