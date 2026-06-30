"""
experiments/efficiency_study.py
================================
Systematic efficiency and accuracy comparison across all schemes.
This is the core of the dissertation's novel contribution.

For each scheme:
  - Run at multiple dt values
  - Compare final state against reference solution (tiny dt)
  - Compute MAE, MSE, L2 error
  - Measure wall-clock time per step
  - Plot error vs dt (convergence) and error vs time (efficiency)

Usage
-----
    python experiments/efficiency_study.py
"""

import sys
import os
import time
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from grid        import Grid
from integrators import step, robert_asselin_filter
from results     import save_experiment

PLOT_DIR = "output/figures"
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Reference solution (very small dt — treated as "exact")
# ---------------------------------------------------------------------------

def compute_reference(bubble_amp=2.0, ref_dt=0.002, t_end=2.0,
                       scheme='RK4'):
    """
    Compute reference solution at very small dt.

    Uses RK4 (not CTCS) so errors for all schemes are measured against
    a scheme-agnostic reference.  A CTCS reference would cancel its own
    temporal error, giving artificially small L2 values when testing CTCS.
    """
    n_steps = int(t_end / ref_dt)
    print(f"  Reference: {scheme}  dt={ref_dt}s  "
          f"steps={n_steps}  T={t_end}s")

    grid      = Grid({"Lx": 1000.0, "Lz": 1000.0,
                      "dx": 10.0,   "dz": 10.0})
    state     = grid.allocate_state()
    state_old = None  # CTCS bootstraps with FTCS on step 0 when None

    xc   = grid.Lx / 2.0
    zc   = grid.Lz * 0.4
    r    = 150.0
    r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
    state["theta"] = bubble_amp * np.exp(-r_sq / bubble_radius**2)

    t = 0.0
    for n in range(n_steps):
        # Save φ^(n-1) BEFORE step() overwrites state_old with φ^n
        state_prev = state_old
        state_new, state_old, _ = step(state, grid, ref_dt,
                                       scheme=scheme,
                                       state_old=state_old)
        if scheme == 'CTCS' and n > 0:
            # Proper Robert-Asselin filter: φ^n_f = φ^n + α*(φ^(n-1) - 2φ^n + φ^(n+1))
            state_filtered = robert_asselin_filter(state_prev, state,
                                                   state_new, alpha=0.1)
            state_old = state_filtered
            state = state_new
        else:
            state = state_new
        t += ref_dt

        if np.any(np.isnan(state['u'])):
            print(f"  Reference BLOW-UP at t={t:.3f}s")
            return None, grid

    print(f"  Reference complete at t={t:.2f}s")
    return state, grid


# ---------------------------------------------------------------------------
# Compute errors vs reference
# ---------------------------------------------------------------------------

def compute_errors(ref, test):
    """MAE, MSE, L2 for each variable."""
    errors = {}
    for var in ['u', 'w', 'theta', 'pi']:
        diff = ref[var] - test[var]
        ref_norm = np.sqrt(np.sum(ref[var]**2))
        errors[var] = {
            'MAE': float(np.mean(np.abs(diff))),
            'MSE': float(np.mean(diff**2)),
            'L2':  float(np.sqrt(np.sum(diff**2)) /
                         (ref_norm + 1e-30)),
        }
    return errors


# ---------------------------------------------------------------------------
# Run one scheme at one dt and return errors + timing
# ---------------------------------------------------------------------------

def run_scheme(scheme, bubble_amp, dt, t_end, ref_state, grid):
    """Run scheme, return (errors, wall_time_per_step). None if blow-up."""
    n_steps = int(t_end / dt)
    state   = grid.allocate_state()
    s_old   = grid.allocate_state()

    xc   = grid.Lx / 2.0
    zc   = grid.Lz * 0.4
    r    = 150.0
    r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
    state["theta"] = bubble_amp * np.exp(-r_sq / bubble_radius**2)

    t_wall = 0.0
    t      = 0.0

    for n in range(n_steps):
        t0 = time.perf_counter()
        # Save φ^(n-1) BEFORE step() overwrites s_old with φ^n
        s_prev = s_old
        try:
            state_new, s_old, _ = step(state, grid, dt,
                                    scheme=scheme,
                                    state_old=s_old)
        except Exception:
            return None, None

        if scheme == 'CTCS' and n > 0:
            # Proper Robert-Asselin filter: φ^n_f = φ^n + α*(φ^(n-1) - 2φ^n + φ^(n+1))
            state_filtered = robert_asselin_filter(s_prev, state,
                                                   state_new, alpha=0.1)
            # For the next leapfrog step: current = φ^(n+1), old = φ^n_filtered
            s_old = state_filtered
            state = state_new
        else:
            state = state_new

        t_wall += time.perf_counter() - t0
        t      += dt

        if np.any(np.isnan(state['u'])) or np.any(np.isinf(state['u'])):
            return None, None

    errors = compute_errors(ref_state, state)
    return errors, t_wall / n_steps   # per-step time


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_efficiency_study(bubble_amp=2.0, t_end=2.0,
                         ref_dt=0.002):
    """Full efficiency and accuracy comparison."""

    schemes  = ['FTCS', 'CTCS', 'RK4', 'SI', 'EPI2', 'EPI3']
    test_dts = [0.005, 0.01, 0.02, 0.05, 0.1]
    colors   = {'FTCS':'red', 'CTCS':'blue', 'RK4':'green',
                'SI':'purple', 'EPI2':'orange', 'EPI3':'brown'}

    # Reference solution
    print("\nComputing reference solution...")
    ref_state, grid = compute_reference(bubble_amp, ref_dt, t_end)
    if ref_state is None:
        print("Reference failed.")
        return

    # Results storage
    # results[scheme][dt] = {errors, time_per_step}
    results = {s: {} for s in schemes}

    print("\nRunning scheme comparisons...")
    for scheme in schemes:
        for dt in test_dts:
            n_steps = int(t_end / dt)
            print(f"  {scheme:<6}  dt={dt:.3f}  steps={n_steps}...",
                  end="", flush=True)
            errs, tps = run_scheme(scheme, bubble_amp, dt,
                                   t_end, ref_state, grid)
            if errs is None:
                print(" BLOW-UP")
            else:
                results[scheme][dt] = {"errors": errs, "time_per_step": tps}
                l2 = errs['theta']['L2']
                print(f" L2(θ')={l2:.3e}  t/step={tps*1000:.2f}ms")

    # Save
    save_experiment(
        name     = "efficiency_study",
        state    = ref_state,
        grid     = grid,
        metadata = {"experiment": "efficiency_study",
                    "ref_dt": ref_dt, "t_end": t_end,
                    "schemes": schemes, "test_dts": test_dts}
    )

    # Plot 1: L2 error vs dt (convergence plot)
    _plot_convergence(results, schemes, colors, t_end)

    # Plot 2: L2 error vs wall-clock time (efficiency plot)
    _plot_efficiency(results, schemes, colors, test_dts, t_end)

    return results


def _plot_convergence(results, schemes, colors, t_end):
    """Log-log plot of L2(theta') vs dt for each scheme."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Convergence — L2 error vs Δt  |  t_end={t_end}s",
                 fontsize=12, fontweight='bold')

    metrics = ['MAE', 'MSE', 'L2']
    for ax, metric in zip(axes, metrics):
        for scheme in schemes:
            dts  = sorted(results[scheme].keys())
            vals = [results[scheme][dt]['errors']['theta'][metric]
                    for dt in dts]
            if dts:
                ax.loglog(dts, vals, 'o-', color=colors[scheme],
                          label=scheme, lw=1.8, markersize=6)

        # Reference slope lines
        if dts:
            da = np.array(dts)
            ref_v = vals[-1] / dts[-1]
            ax.loglog(da, da * ref_v, 'k--', alpha=0.4,
                      lw=1, label='1st order')
            ax.loglog(da, da**2 * ref_v / dts[-1], 'k:',
                      alpha=0.4, lw=1, label='2nd order')

        ax.set_xlabel("Δt  (s)", fontsize=10)
        ax.set_ylabel(f"{metric}(θ')", fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "convergence_plot.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"\n  Saved: {fname}")
    plt.close(fig)


def _plot_efficiency(results, schemes, colors, test_dts, t_end):
    """L2 error vs total wall-clock time — the efficiency frontier."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Efficiency plot — L2(θ') vs wall-clock time\n"
        "Bottom-left corner = best (accurate AND fast)",
        fontsize=11, fontweight='bold'
    )

    for scheme in schemes:
        dts   = sorted(results[scheme].keys())
        l2s   = [results[scheme][dt]['errors']['theta']['L2']
                 for dt in dts]
        n_steps_list = [int(t_end / dt) for dt in dts]
        times = [results[scheme][dt]['time_per_step'] * n
                 for dt, n in zip(dts, n_steps_list)]

        if dts:
            ax.loglog(times, l2s, 'o-', color=colors[scheme],
                      label=scheme, lw=2, markersize=7)
            # Annotate dt values
            for t_w, l2, dt in zip(times, l2s, dts):
                ax.annotate(f"Δt={dt}", (t_w, l2),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel("Total wall-clock time (s)", fontsize=11)
    ax.set_ylabel("L2 error in θ'", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    fname = os.path.join(PLOT_DIR, "efficiency_plot.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    run_efficiency_study(bubble_amp=2.0, t_end=2.0, ref_dt=0.002)
