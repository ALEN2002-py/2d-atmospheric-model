"""
compare_schemes.py
==================
Standalone accuracy and efficiency comparison across all time schemes.

This is the core of the dissertation's novel contribution:
  "Systematic comparison of RK4, semi-implicit, EPI2, EPI3"

What this script does:
  1. Computes a reference solution using CTCS at very small dt
  2. Runs each scheme at several dt values
  3. Computes MAE, MSE, L2 error vs reference for each variable
  4. Measures wall-clock time per step
  5. Saves all results to output/results/
  6. Produces four plots:
       - Convergence plot: L2 error vs dt (log-log)
       - Error heatmap:    L2 error for all schemes x all dt
       - Efficiency plot:  L2 error vs wall-clock time
       - Time series:      max|theta'| over time for each scheme

Usage
-----
    python compare_schemes.py                        # all schemes, defaults
    python compare_schemes.py --t_end 4.0            # longer run
    python compare_schemes.py --schemes CTCS RK4     # subset of schemes
"""

import argparse
import os
import sys
import time
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from grid        import Grid
from integrators import step, robert_asselin_filter
from results     import save_experiment

# Output directories
PLOT_DIR    = "output/figures"
RESULTS_DIR = "output/results"
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Constants
# ===========================================================================

ALL_SCHEMES = ['FTCS', 'BTCS', 'CTCS', 'RK4', 'SI', 'EPI2', 'EPI3']

SCHEME_COLORS = {
    'FTCS': '#e74c3c',
    'BTCS': '#e67e22',
    'CTCS': '#3498db',
    'RK4':  '#2ecc71',
    'SI':   '#9b59b6',
    'EPI2': '#1abc9c',
    'EPI3': '#f39c12',
}

SCHEME_ORDER = {s: i for i, s in enumerate(ALL_SCHEMES)}


# ===========================================================================
# Helper: set up initial conditions on a grid
# ===========================================================================

def make_initial_state(grid, bubble_amp=2.0):
    """
    Set up a Gaussian warm bubble initial condition.

    theta'(x,z) = bubble_amp * exp(-r^2 / bubble_radius^2)

    The bubble is centred at (Lx/2, 0.4*Lz) with e-folding radius
    bubble_radius = 150 m. This is a generic test bubble (not the exact
    G&R cosine bell); it is used by compare_schemes, efficiency_study,
    warm_bubble, and menu.py for quick scheme comparisons.

    Parameters
    ----------
    grid       : Grid
    bubble_amp : float  — peak theta' perturbation [K]  (default 2.0)

    Returns
    -------
    state : dict  — {u, w, theta, pi}, theta is the only non-zero field
    """
    state = grid.allocate_state()
    if bubble_amp > 0:
        xc            = grid.Lx / 2.0   # horizontal centre [m]
        zc            = grid.Lz * 0.4   # vertical centre [m] — lower than mid-domain
        bubble_radius = 150.0           # e-folding radius [m]
        r_sq          = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
        state["theta"] = bubble_amp * np.exp(-r_sq / bubble_radius**2)
    return state


# ===========================================================================
# Run one simulation, return final state + timing
# ===========================================================================

def run_simulation(scheme, grid, bubble_amp, dt, t_end,
                   collect_series=False):
    """
    Run scheme from t=0 to t_end.

    Returns
    -------
    final_state  : dict or None (None if blow-up)
    time_per_step: float [seconds]
    series       : list of (t, max_theta) — only if collect_series=True
    """
    n_steps   = int(round(t_end / dt))
    state     = make_initial_state(grid, bubble_amp)
    state_old = None  # CTCS bootstraps with FTCS on step 0 when None

    total_wall = 0.0
    series     = [] if collect_series else None
    t          = 0.0

    for n in range(n_steps):
        t0 = time.perf_counter()

        # Save φ^(n-1) BEFORE step() overwrites state_old with φ^n
        state_prev = state_old
        try:
            state_new, state_old, _ = step(state, grid, dt,
                                           scheme=scheme,
                                           state_old=state_old)
        except Exception as e:
            print(f"    [{scheme}] Exception at step {n+1}: {e}")
            return None, None, series

        if scheme == 'CTCS' and n > 0:
            # Proper Robert-Asselin filter: φ^n_f = φ^n + α*(φ^(n-1) - 2φ^n + φ^(n+1))
            state_filtered = robert_asselin_filter(state_prev, state,
                                                   state_new, alpha=0.1)
            # For the next leapfrog step: current = φ^(n+1), old = φ^n_filtered
            state_old = state_filtered
            state = state_new
        else:
            state = state_new

        total_wall += time.perf_counter() - t0
        t          += dt

        # Blow-up check
        if np.any(~np.isfinite(state['u'])):
            return None, None, series

        if collect_series and (n + 1) % max(1, n_steps // 50) == 0:
            series.append((t,
                           float(np.max(np.abs(state['theta']))),
                           float(np.max(np.abs(state['w'])))))

    tps = total_wall / n_steps if n_steps > 0 else 0.0
    return state, tps, series


# ===========================================================================
# Error metrics
# ===========================================================================

def compute_errors(ref, test):
    """
    Compute MAE, MSE, L2 relative error for each variable.
    Returns dict: errors[var] = {MAE, MSE, L2}
    """
    errors = {}
    for var in ['u', 'w', 'theta', 'pi']:
        diff     = ref[var] - test[var]
        ref_norm = np.sqrt(np.sum(ref[var]**2)) + 1e-30
        errors[var] = {
            'MAE': float(np.mean(np.abs(diff))),
            'MSE': float(np.mean(diff**2)),
            'L2':  float(np.sqrt(np.sum(diff**2)) / ref_norm),
        }
    return errors


# ===========================================================================
# Main comparison
# ===========================================================================

def run_comparison(schemes=None, bubble_amp=2.0, t_end=2.0,
                   ref_dt=0.002, test_dts=None):
    """
    Full accuracy and efficiency comparison.

    Parameters
    ----------
    schemes   : list of scheme names to test (default: all)
    bubble_amp: warm bubble amplitude [K]
    t_end     : simulation end time [s]
    ref_dt    : time step for reference solution (very small)
    test_dts  : list of dt values to test
    """
    if schemes is None:
        schemes = ALL_SCHEMES
    if test_dts is None:
        test_dts = [0.005, 0.01, 0.02, 0.05]

    grid = Grid({"Lx": 1000.0, "Lz": 1000.0, "dx": 10.0, "dz": 10.0})

    # ------------------------------------------------------------------
    # Step 1: Reference solution
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("  STEP 1 — Computing reference solution")
    print(f"  scheme=RK4  dt={ref_dt}s  t_end={t_end}s")
    print("  (RK4 reference is scheme-agnostic; CTCS would give")
    print("   artificially small errors for itself)")
    print("="*55)

    ref_state, _, _ = run_simulation('RK4', grid, bubble_amp,
                                     ref_dt, t_end)
    if ref_state is None:
        print("  Reference solution failed. Exiting.")
        return

    print(f"  Reference complete.")
    print(f"  |theta'|_max = {np.max(np.abs(ref_state['theta'])):.4e}")
    print(f"  |w|_max      = {np.max(np.abs(ref_state['w'])):.4e}")

    # ------------------------------------------------------------------
    # Step 2: Run each scheme at each dt
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("  STEP 2 — Running scheme comparisons")
    print("="*55)

    # results[scheme][dt] = {errors, time_per_step}
    results = {s: {} for s in schemes}

    for scheme in schemes:
        print(f"\n  {scheme}")
        hdr = "L2(theta')"
        print(f"  {'dt':>8}  {'steps':>6}  {hdr:>10}  "
              f"{'MAE':>10}  {'ms/step':>8}")
        print("  " + "-"*50)

        for dt in test_dts:
            n_steps = int(round(t_end / dt))
            state, tps, _ = run_simulation(scheme, grid, bubble_amp,
                                           dt, t_end)
            if state is None:
                print(f"  {dt:>8.3f}  {n_steps:>6}  "
                      f"{'BLOW-UP':>10}")
                continue

            errs = compute_errors(ref_state, state)
            results[scheme][dt] = {
                'errors':        errs,
                'time_per_step': tps,
                'n_steps':       n_steps,
            }

            l2  = errs['theta']['L2']
            mae = errs['theta']['MAE']
            print(f"  {dt:>8.3f}  {n_steps:>6}  "
                  f"{l2:>10.3e}  {mae:>10.3e}  "
                  f"{tps*1000:>8.3f}")

    # ------------------------------------------------------------------
    # Step 3: Time series (collect for a fixed dt)
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("  STEP 3 — Collecting time series")
    print("="*55)

    # Use dt=0.02 so schemes differ visibly (CTCS L2≈3e-5 vs RK4 L2≈3e-11).
    # At dt=0.01 all schemes agree to <1e-7 and curves completely overlap.
    series_dt   = test_dts[2] if len(test_dts) > 2 else test_dts[-1]
    series_data = {}

    for scheme in schemes:
        print(f"  {scheme}  dt={series_dt}...", end="", flush=True)
        _, _, ser = run_simulation(scheme, grid, bubble_amp,
                                   series_dt, t_end,
                                   collect_series=True)
        if ser:
            series_data[scheme] = ser
            print(f" {len(ser)} points")
        else:
            print(" BLOW-UP")

    # ------------------------------------------------------------------
    # Step 4: Save results
    # ------------------------------------------------------------------
    # Build a flat metadata dict for saving
    meta = {
        "experiment":  "compare_schemes",
        "schemes":     schemes,
        "test_dts":    test_dts,
        "ref_dt":      ref_dt,
        "t_end":       t_end,
        "bubble_amp":  bubble_amp,
    }
    # Add L2 errors to metadata
    for scheme in schemes:
        for dt in test_dts:
            if dt in results[scheme]:
                key = f"{scheme}_dt{dt}_L2_theta"
                meta[key] = results[scheme][dt]['errors']['theta']['L2']

    save_experiment("compare_schemes", ref_state, grid, metadata=meta)
    print("\n  Results saved to output/results/")

    # ------------------------------------------------------------------
    # Step 5: Plots
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("  STEP 4 — Generating plots")
    print("="*55)

    _plot_convergence(results, schemes, test_dts, t_end)
    _plot_error_heatmap(results, schemes, test_dts)
    _plot_efficiency(results, schemes, test_dts, t_end)
    _plot_time_series(series_data, schemes, series_dt)

    print("\n  All plots saved to output/figures/")
    return results


# ===========================================================================
# Plot 1 — Convergence: L2 error vs dt (log-log)
# ===========================================================================

def _plot_convergence(results, schemes, test_dts, t_end):
    """
    Classic convergence plot.
    Each scheme should show a straight line on log-log axes with
    slope = order of accuracy.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"Convergence Plot — L2 error vs Δt  (t_end = {t_end} s, ref: RK4)\n"
        "Slope of line = order of accuracy  |  flat lines = spatial error floor",
        fontsize=12, fontweight='bold'
    )

    metrics = ['L2', 'MAE', 'MSE']
    m_labels = ['L2 relative error', 'MAE', 'MSE']

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        for scheme in schemes:
            dts  = sorted(results[scheme].keys())
            vals = [results[scheme][dt]['errors']['theta'][metric]
                    for dt in dts]
            if len(dts) >= 2:
                ax.loglog(dts, vals, 'o-',
                          color=SCHEME_COLORS.get(scheme, 'black'),
                          label=scheme, lw=2, markersize=7)

        # Reference order lines — anchor at median error at largest dt
        if dts:
            da   = np.array(sorted(test_dts))
            base = da[-1]
            # Find a representative error magnitude at largest dt
            ref_vals = [results[s][base]['errors']['theta'][metric]
                        for s in schemes
                        if base in results[s]
                        and np.isfinite(results[s][base]['errors']['theta'][metric])]
            anchor = float(np.median(ref_vals)) if ref_vals else 1e-3
            # 1st order
            ax.loglog(da, (da/base)**1 * anchor, 'k--',
                      alpha=0.45, lw=1.4, label='1st order')
            # 2nd order
            ax.loglog(da, (da/base)**2 * anchor, 'k:',
                      alpha=0.45, lw=1.4, label='2nd order')
            # 4th order
            ax.loglog(da, (da/base)**4 * anchor, 'k-.',
                      alpha=0.45, lw=1.4, label='4th order')

        ax.set_xlabel("Δt  (s)", fontsize=10)
        ax.set_ylabel(f"{mlabel} of θ'", fontsize=10)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "convergence_plot.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


# ===========================================================================
# Plot 2 — Error heatmap: schemes x dt
# ===========================================================================

def _plot_error_heatmap(results, schemes, test_dts):
    """
    Heatmap showing L2(theta') for every (scheme, dt) combination.
    Green = small error (good), Red = large error (bad).
    Grey = blow-up / not run.
    """
    dts_sorted = sorted(test_dts)
    n_s        = len(schemes)
    n_d        = len(dts_sorted)

    # Build matrix
    mat = np.full((n_s, n_d), np.nan)
    for i, scheme in enumerate(schemes):
        for j, dt in enumerate(dts_sorted):
            if dt in results[scheme]:
                mat[i, j] = results[scheme][dt]['errors']['theta']['L2']

    fig, ax = plt.subplots(figsize=(max(6, n_d * 1.4),
                                    max(4, n_s * 0.8)))
    fig.suptitle(
        "Error Heatmap — L2(θ') for each scheme × Δt\n"
        "Green = accurate  |  Red = large error  |  Grey = blow-up",
        fontsize=11, fontweight='bold'
    )

    # Use log scale for colour
    log_mat = np.where(np.isnan(mat), np.nan, np.log10(mat + 1e-30))
    vmin    = np.nanmin(log_mat)
    vmax    = np.nanmax(log_mat)

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color='#cccccc')   # grey for blow-up

    im = ax.imshow(log_mat, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto')

    # Axis labels
    ax.set_xticks(range(n_d))
    ax.set_xticklabels([f"Δt={dt}" for dt in dts_sorted], fontsize=9)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(schemes, fontsize=10)
    ax.set_xlabel("Time step Δt", fontsize=10)
    ax.set_ylabel("Scheme", fontsize=10)

    # Annotate cells with actual values
    for i in range(n_s):
        for j in range(n_d):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2e}",
                        ha='center', va='center',
                        fontsize=7, color='black')
            else:
                ax.text(j, i, "BLOW-UP",
                        ha='center', va='center',
                        fontsize=7, color='#888888')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10(L2 error)", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "error_heatmap.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


# ===========================================================================
# Plot 3 — Efficiency: L2 error vs total wall-clock time
# ===========================================================================

def _plot_efficiency(results, schemes, test_dts, t_end):
    """
    Efficiency frontier plot.
    x-axis: total wall-clock time for the run
    y-axis: L2 error vs reference

    The BEST scheme sits in the bottom-left corner:
    low error AND fast.

    This is the key plot for the dissertation comparison.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(
        "Efficiency Plot — L2(θ') vs Total Wall-Clock Time\n"
        "Bottom-left = best (accurate AND fast)",
        fontsize=12, fontweight='bold'
    )

    for scheme in schemes:
        dts   = sorted(results[scheme].keys())
        l2s   = [results[scheme][dt]['errors']['theta']['L2']
                 for dt in dts]
        times = [results[scheme][dt]['time_per_step'] *
                 results[scheme][dt]['n_steps']
                 for dt in dts]

        if dts:
            ax.loglog(times, l2s, 'o-',
                      color=SCHEME_COLORS.get(scheme, 'black'),
                      label=scheme, lw=2.5, markersize=9,
                      markeredgecolor='white', markeredgewidth=0.8)
            # Label each point with its dt
            for t_w, l2, dt in zip(times, l2s, dts):
                ax.annotate(f"Δt={dt}",
                            xy=(t_w, l2),
                            xytext=(5, 3),
                            textcoords='offset points',
                            fontsize=7, alpha=0.8,
                            color=SCHEME_COLORS.get(scheme, 'black'))

    ax.set_xlabel("Total wall-clock time  (s)", fontsize=11)
    ax.set_ylabel("L2 relative error in θ'", fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')

    # Annotate ideal region
    ax.annotate("← Ideal region\n   (fast + accurate)",
                xy=(0.05, 0.08), xycoords='axes fraction',
                fontsize=9, color='green', alpha=0.7)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "efficiency_plot.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


# ===========================================================================
# Plot 4 — Time series: max|theta'| over time for each scheme
# ===========================================================================

def _plot_time_series(series_data, schemes, dt):
    """
    Two-panel time series:
      Left  — max|θ'|: warm bubble amplitude (RA filter damping visible)
      Right — max|w|:  convective updraft strength (grows from 0)

    Using a larger dt (0.02) ensures CTCS near its stability limit shows
    visibly different evolution from RK4/SI, making scheme differences
    discernible. At dt=0.01 all curves are indistinguishable (<1e-7 error).
    """
    if not series_data:
        print("  No time series data collected.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Time Series — Δt = {dt} s  |  "
        "Curves that stop early = scheme blow-up at this dt\n"
        "max|θ'| shows amplitude damping; max|w| shows convective growth",
        fontsize=11, fontweight='bold'
    )

    ax_theta, ax_w = axes

    for scheme, series in series_data.items():
        t_vals    = [s[0] for s in series]
        theta_max = [s[1] for s in series]
        # Series entries may be 2-tuple (old) or 3-tuple (new); handle both
        w_max     = [s[2] for s in series] if len(series[0]) > 2 else None
        color = SCHEME_COLORS.get(scheme, 'black')
        ax_theta.plot(t_vals, theta_max, color=color, label=scheme, lw=2)
        if w_max is not None:
            ax_w.plot(t_vals, w_max, color=color, label=scheme, lw=2)

    ax_theta.set_xlabel("Time  (s)", fontsize=11)
    ax_theta.set_ylabel("max |θ'|  (K)", fontsize=11)
    ax_theta.set_title("Bubble amplitude (θ' perturbation)", fontsize=11)
    ax_theta.legend(fontsize=10)
    ax_theta.grid(True, alpha=0.3)

    ax_w.set_xlabel("Time  (s)", fontsize=11)
    ax_w.set_ylabel("max |w|  (m s⁻¹)", fontsize=11)
    ax_w.set_title("Convective updraft (w grows from 0 by buoyancy)", fontsize=11)
    ax_w.legend(fontsize=10)
    ax_w.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "time_series_theta.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Accuracy and efficiency comparison of time schemes"
    )
    parser.add_argument(
        "--schemes", nargs="+", default=None,
        choices=ALL_SCHEMES,
        help="Schemes to compare (default: all)"
    )
    parser.add_argument(
        "--t_end", type=float, default=2.0,
        help="Simulation end time in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--ref_dt", type=float, default=0.002,
        help="Reference solution time step (default: 0.002)"
    )
    parser.add_argument(
        "--bubble_amp", type=float, default=2.0,
        help="Warm bubble amplitude in K (default: 2.0)"
    )
    parser.add_argument(
        "--test_dts", nargs="+", type=float,
        default=[0.005, 0.01, 0.02, 0.05],
        help="dt values to test (default: 0.005 0.01 0.02 0.05)"
    )
    args = parser.parse_args()

    run_comparison(
        schemes    = args.schemes,
        bubble_amp = args.bubble_amp,
        t_end      = args.t_end,
        ref_dt     = args.ref_dt,
        test_dts   = args.test_dts,
    )
