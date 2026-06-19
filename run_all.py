"""
run_all.py
==========
Runs EVERYTHING in one go:
  1. Zero amplitude test — all 7 schemes
  2. Warm bubble — contour plots (theta' and w)
  3. Full scheme comparison — convergence, error heatmap,
     efficiency plot, time series

All outputs saved to:
  output/results/   — .npz + .json data
  output/figures/   — all plots

Usage
-----
    python run_all.py                 # default settings (~2-5 min)
    python run_all.py --quick         # faster, smaller t_end (~30s)
    python run_all.py --t_end 4.0     # longer comparison run
"""

import argparse
import os
import sys
import time

sys.path.insert(0, "src")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid import Grid
from integrators import step, robert_asselin_filter
from results import save_experiment
from compare_schemes import run_comparison, ALL_SCHEMES

PLOT_DIR = "output/figures"
RESULTS_DIR = "output/results"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# 1. Zero amplitude test
# ===========================================================================

def run_zero_amplitude_test(dt=0.02, n_steps=100):
    print("\n" + "=" * 60)
    print("  1. ZERO AMPLITUDE TEST — all 7 schemes")
    print("=" * 60)

    grid = Grid(
        {
            "Lx": 1000.0,
            "Lz": 1000.0,
            "dx": 10.0,
            "dz": 10.0,
        }
    )

    schemes = ALL_SCHEMES

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    fig.suptitle(
        f"Zero Amplitude Test  |  dt={dt}s  n_steps={n_steps}\n"
        "All lines should stay at zero",
        fontsize=12,
        fontweight="bold",
    )

    vars_ = ["u", "w", "theta", "pi"]
    labels = ["|u|", "|w|", "|θ'|", "|π'|"]

    colors = plt.cm.tab10(np.linspace(0, 1, len(schemes)))

    for scheme, color in zip(schemes, colors):
        state = grid.allocate_state()
        state_old = None

        times = [0.0]
        series = {v: [0.0] for v in vars_}

        t = 0.0

        for n in range(n_steps):
            state_new, state_old = step(
                state,
                grid,
                dt,
                scheme=scheme,
                state_old=state_old,
            )

            if scheme == "CTCS" and n > 0:
                state = robert_asselin_filter(
                    state_old,
                    state,
                    state_new,
                    alpha=0.1,
                )
            else:
                state = state_new

            t += dt
            times.append(t)

            for v in vars_:
                series[v].append(float(np.max(np.abs(state[v]))))

        max_val = max(max(series[v]) for v in vars_)

        status = (
            "PASS"
            if max_val < 1e-10
            else f"FAIL ({max_val:.1e})"
        )

        print(
            f"  {scheme:6s}  max|all vars| = "
            f"{max_val:.2e}   {status}"
        )

        for ax, v, label in zip(axes, vars_, labels):
            ax.plot(
                times,
                series[v],
                color=color,
                label=scheme,
                lw=1.3,
            )

    for ax, label in zip(axes, labels):
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(f"max {label}", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fname = os.path.join(
        PLOT_DIR,
        "zero_amplitude_test.png",
    )

    plt.savefig(
        fname,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  Saved: {fname}")


# ===========================================================================
# 2. Warm bubble — contour plots
# ===========================================================================

def run_warm_bubble(
    scheme="RK4",
    bubble_amp=2.0,
    dt=0.02,
    n_steps=500,
):
    print("\n" + "=" * 60)
    print(f"  2. WARM BUBBLE — {scheme}")
    print("=" * 60)

    grid = Grid(
        {
            "Lx": 1000.0,
            "Lz": 1000.0,
            "dx": 10.0,
            "dz": 10.0,
        }
    )

    state = grid.allocate_state()
    state_old = None

    xc = grid.Lx / 2
    zc = grid.Lz * 0.4
    r = 150.0

    r2 = (
        (grid.x_2d - xc) ** 2
        + (grid.z_2d - zc) ** 2
    )

    state["theta"] = bubble_amp * np.exp(-r2 / r**2)

    save_at = set(
        int(n_steps * f)
        for f in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    save_at.add(n_steps)

    snapshots = [
        (
            0.0,
            {k: v.copy() for k, v in state.items()},
        )
    ]

    t = 0.0

    for n in range(n_steps):
        state_new, state_old = step(
            state,
            grid,
            dt,
            scheme=scheme,
            state_old=state_old,
        )

        if scheme == "CTCS" and n > 0:
            state = robert_asselin_filter(
                state_old,
                state,
                state_new,
                alpha=0.1,
            )
        else:
            state = state_new

        t += dt

        if np.any(~np.isfinite(state["u"])):
            print(f"  BLOW-UP at t={t:.2f}s")
            break

        if (n + 1) in save_at:
            snapshots.append(
                (
                    t,
                    {
                        k: v.copy()
                        for k, v in state.items()
                    },
                )
            )

    print(
        f"  Complete. "
        f"|w|_max={np.max(np.abs(state['w'])):.4e}  "
        f"|theta'|_max={np.max(np.abs(state['theta'])):.4f}"
    )

    save_experiment(
        f"warm_bubble_{scheme}",
        state,
        grid,
        snapshots=snapshots,
        metadata={
            "scheme": scheme,
            "dt": dt,
            "n_steps": n_steps,
            "bubble_amp": bubble_amp,
        },
    )

    _contour_plot(
        snapshots,
        grid,
        "theta",
        scheme,
        bubble_amp,
        dt,
    )

    _contour_plot(
        snapshots,
        grid,
        "w",
        scheme,
        bubble_amp,
        dt,
    )


def _contour_plot(
    snapshots,
    grid,
    variable,
    scheme,
    bubble_amp,
    dt,
):
    x_km = grid.x_1d / 1000
    z_km = grid.z_1d / 1000

    X, Z = np.meshgrid(x_km, z_km)

    n = len(snapshots)

    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    all_vals = np.concatenate(
        [
            s[variable].ravel()
            for _, s in snapshots
        ]
    )

    vmax = max(
        np.max(np.abs(all_vals)),
        1e-10,
    )

    if variable == "theta":
        cmap = "RdYlBu_r"
        levels_fill = np.linspace(0, vmax, 20)
        levels_line = np.linspace(
            0.1 * vmax,
            vmax,
            7,
        )
        cbar_label = "θ' (K)"
        vmin = 0.0
    else:
        cmap = "RdBu_r"
        levels_fill = np.linspace(
            -vmax,
            vmax,
            21,
        )
        levels_line = np.linspace(
            0.1 * vmax,
            vmax,
            6,
        )
        cbar_label = "w (m/s)"
        vmin = -vmax

    for idx, (t, state) in enumerate(snapshots):
        ax = axes[idx // n_cols][idx % n_cols]

        data = state[variable]

        cf = ax.contourf(
            X,
            Z,
            data,
            levels=levels_fill,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.88,
        )

        cl = ax.contour(
            X,
            Z,
            data,
            levels=levels_line,
            colors="black",
            linewidths=0.7,
            alpha=0.7,
        )

        ax.clabel(
            cl,
            fmt="%.2g",
            fontsize=6,
            inline=True,
        )

        if variable == "w":
            ax.contour(
                X,
                Z,
                data,
                levels=np.linspace(
                    -vmax,
                    -0.1 * vmax,
                    6,
                ),
                colors="navy",
                linewidths=0.7,
                linestyles="dashed",
                alpha=0.7,
            )

        ax.axhline(
            0.8,
            color="steelblue",
            lw=0.8,
            linestyle="--",
            alpha=0.5,
        )

        ax.set_title(
            f"t = {t:.1f} s",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel("x (km)", fontsize=9)
        ax.set_ylabel("z (km)", fontsize=9)

        ax.set_aspect("equal")

        fig.colorbar(
            cf,
            ax=ax,
            fraction=0.046,
            pad=0.03,
            label=cbar_label,
        )

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    var_label = "θ'" if variable == "theta" else "w"

    fig.suptitle(
        f"Warm Bubble — {var_label}  |  "
        f"{scheme}  |  A={bubble_amp}K",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()

    fname = os.path.join(
        PLOT_DIR,
        f"warm_bubble_{scheme}_{variable}.png",
    )

    plt.savefig(
        fname,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  Saved: {fname}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast settings for a quick check (~30s)",
    )

    p.add_argument(
        "--t_end",
        type=float,
        default=2.0,
        help="Comparison run length [s]",
    )

    p.add_argument(
        "--bubble_dt",
        type=float,
        default=0.02,
    )

    p.add_argument(
        "--bubble_steps",
        type=int,
        default=500,
    )

    args = p.parse_args()

    start = time.time()

    if args.quick:
        zero_steps = 50
        t_end = 0.5
        bubble_steps = 100
        test_dts = [0.01, 0.02]
        ref_dt = 0.005
        schemes = [
            "FTCS",
            "CTCS",
            "RK4",
            "SI",
            "EPI2",
            "EPI3",
        ]
    else:
        zero_steps = 100
        t_end = args.t_end
        bubble_steps = args.bubble_steps
        test_dts = [
            0.005,
            0.01,
            0.02,
            0.05,
        ]
        ref_dt = 0.002
        schemes = ALL_SCHEMES

    run_zero_amplitude_test(
        dt=0.02,
        n_steps=zero_steps,
    )

    run_warm_bubble(
        scheme="RK4",
        bubble_amp=2.0,
        dt=args.bubble_dt,
        n_steps=bubble_steps,
    )

    run_warm_bubble(
        scheme="CTCS",
        bubble_amp=2.0,
        dt=args.bubble_dt,
        n_steps=bubble_steps,
    )

    print("\n" + "=" * 60)
    print("  3. SCHEME COMPARISON")
    print("  (convergence/efficiency/heatmap)")
    print("=" * 60)

    run_comparison(
        schemes=schemes,
        bubble_amp=2.0,
        t_end=t_end,
        ref_dt=ref_dt,
        test_dts=test_dts,
    )

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"  ALL DONE in {elapsed:.1f}s")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Figures: {PLOT_DIR}/")
    print("=" * 60)

    print("\n  Generated figures:")
    for f in sorted(os.listdir(PLOT_DIR)):
        print(f"    {f}")

    print("\n  Generated result files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".json"):
            print(f"    {f}")