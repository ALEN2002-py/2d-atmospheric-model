"""
plot_results.py
===============
All visualisation functions for the atmospheric model.

Plots produced:
  1. plot_warm_bubble()     — 2D heatmap of theta' at multiple times
  2. plot_velocity()        — 2D heatmap of w (vertical velocity)
  3. plot_zero_amp_test()   — time series showing all vars stay at zero
  4. plot_error_comparison() — MAE/MSE/L2 error vs reference solution
  5. plot_scheme_comparison() — side-by-side theta' for all schemes

Usage
-----
    python plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.insert(0, "src")

from grid        import Grid
from integrators import step, robert_asselin_filter

# Output directory for plots
PLOT_DIR = "output/figures"
os.makedirs(PLOT_DIR, exist_ok=True)


# ===========================================================================
# Helper: run model and collect snapshots
# ===========================================================================

def run_and_snap(scheme, bubble_amp, dt, n_steps,
                 save_every=None, silent=False):
    """
    Run model, return list of (time, state) snapshots.
    save_every: save snapshot every N steps. None = only save final state.
    """
    grid      = Grid({"Lx": 1000.0, "Lz": 1000.0,
                      "dx": 10.0,   "dz": 10.0})
    state     = grid.allocate_state()
    state_old = grid.allocate_state()

    # Initial condition
    if bubble_amp > 0:
        xc   = grid.Lx / 2.0
        zc   = grid.Lz * 0.4
        r    = 150.0
        r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
        state["theta"] = bubble_amp * np.exp(-r_sq / r**2)

    # Always save t=0
    snapshots = [(0.0, {k: v.copy() for k, v in state.items()})]

    t        = 0.0
    blown_up = False

    for n in range(n_steps):
        state_new, state_old = step(state, grid, dt,
                                    scheme=scheme,
                                    state_old=state_old)
        if scheme == 'CTCS' and n > 0:
            state = robert_asselin_filter(state_old, state,
                                          state_new, alpha=0.1)
        else:
            state = state_new

        t += dt

        # Blow-up check
        if np.any(np.isnan(state['u'])) or np.any(np.isinf(state['u'])):
            if not silent:
                print(f"  [{scheme}] Blow-up at step {n+1}, t={t:.2f}s")
            blown_up = True
            break

        if save_every and (n + 1) % save_every == 0:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

    if not blown_up:
        # Always save final state
        if not snapshots or snapshots[-1][0] != t:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

    return snapshots, grid, blown_up


# ===========================================================================
# Plot 1: Warm Bubble — 2D heatmap of theta' at multiple times
# ===========================================================================

def plot_warm_bubble(scheme='CTCS', bubble_amp=2.0, dt=0.02,
                     n_steps=500, n_frames=5):
    """
    Plot 2D heatmaps of theta' (temperature perturbation) at evenly
    spaced times. Shows the warm bubble rising through the domain.
    """
    save_every = max(1, n_steps // (n_frames - 1))
    snaps, grid, blown = run_and_snap(scheme, bubble_amp, dt,
                                      n_steps, save_every=save_every)

    print(f"  [{scheme}] {len(snaps)} snapshots collected")

    fig, axes = plt.subplots(1, len(snaps),
                             figsize=(3.5 * len(snaps), 4))
    if len(snaps) == 1:
        axes = [axes]

    fig.suptitle(
        f"Warm bubble — θ' (K)  |  scheme: {scheme}  |  "
        f"A={bubble_amp}K  dt={dt}s",
        fontsize=12, fontweight='bold'
    )

    # Consistent colour scale across all frames
    all_theta = np.concatenate([s["theta"].ravel() for _, s in snaps])
    vmax = np.max(np.abs(all_theta))
    vmin = 0.0

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0

    for ax, (t, state) in zip(axes, snaps):
        im = ax.pcolormesh(x_km, z_km, state["theta"],
                           cmap='hot_r', vmin=vmin, vmax=vmax,
                           shading='auto')
        ax.set_title(f"t = {t:.1f} s", fontsize=10)
        ax.set_xlabel("x (km)", fontsize=8)
        ax.set_aspect('equal')
        ax.axhline(0.8, color='dodgerblue', lw=0.8,
                   linestyle='--', alpha=0.7, label='sponge')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("z (km)", fontsize=8)
    plt.tight_layout()

    fname = os.path.join(PLOT_DIR, f"warm_bubble_{scheme}.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.show()


# ===========================================================================
# Plot 2: Vertical velocity w — shows bubble rising
# ===========================================================================

def plot_vertical_velocity(scheme='CTCS', bubble_amp=2.0,
                           dt=0.02, n_steps=500, n_frames=5):
    """
    Plot w (vertical velocity) heatmaps. Positive = upward motion.
    Blue-red diverging colourmap: red = rising, blue = sinking.
    """
    save_every = max(1, n_steps // (n_frames - 1))
    snaps, grid, _ = run_and_snap(scheme, bubble_amp, dt,
                                  n_steps, save_every=save_every)

    fig, axes = plt.subplots(1, len(snaps),
                             figsize=(3.5 * len(snaps), 4))
    if len(snaps) == 1:
        axes = [axes]

    fig.suptitle(
        f"Vertical velocity w (m/s)  |  scheme: {scheme}",
        fontsize=12, fontweight='bold'
    )

    all_w  = np.concatenate([s["w"].ravel() for _, s in snaps])
    vmax   = max(np.max(np.abs(all_w)), 1e-10)
    x_km   = grid.x_1d / 1000.0
    z_km   = grid.z_1d / 1000.0

    for ax, (t, state) in zip(axes, snaps):
        im = ax.pcolormesh(x_km, z_km, state["w"],
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           shading='auto')
        ax.set_title(f"t = {t:.1f} s", fontsize=10)
        ax.set_xlabel("x (km)", fontsize=8)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("z (km)", fontsize=8)
    plt.tight_layout()

    fname = os.path.join(PLOT_DIR, f"vertical_velocity_{scheme}.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.show()


# ===========================================================================
# Plot 3: Zero amplitude test — time series for all schemes
# ===========================================================================

def plot_zero_amp_test(dt=0.02, n_steps=200):
    """
    Run zero amplitude test for all schemes and plot max of each
    variable over time. All lines should stay at or near zero.
    Shows which schemes remain stable and which don't.
    """
    schemes = ['FTCS', 'BTCS', 'CTCS', 'RK4']
    colors  = ['red', 'orange', 'blue', 'green']
    vars_   = ['u', 'w', 'theta', 'pi']
    labels  = ['|u|', '|w|', "|θ'|", "|π'|"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    fig.suptitle(
        f"Zero Amplitude Test — max|var| over time\n"
        f"dt={dt}s  n_steps={n_steps}  "
        f"(all lines should stay at zero)",
        fontsize=12, fontweight='bold'
    )

    # Storage: times_data[scheme][var] = list of max values
    for scheme, color in zip(schemes, colors):
        print(f"  Running zero amp test: {scheme}...")
        snaps, grid, blown = run_and_snap(
            scheme, bubble_amp=0.0, dt=dt,
            n_steps=n_steps, save_every=1, silent=True
        )

        times = np.array([t for t, _ in snaps])

        for ax, var, label in zip(axes, vars_, labels):
            maxvals = np.array([np.max(np.abs(s[var])) for _, s in snaps])
            ax.plot(times, maxvals, color=color,
                    label=scheme, linewidth=1.5)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(f"max {label}", fontsize=9)
            ax.set_title(label, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "zero_amp_test_all_schemes.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.show()


# ===========================================================================
# Plot 4: Accuracy comparison — error vs reference solution
# ===========================================================================

def compute_errors(state_ref, state_test):
    """
    Compute MAE, MSE, L2 norm for each variable.
    Reference is the 'true' solution (very small dt).
    """
    errors = {}
    for var in ['u', 'w', 'theta', 'pi']:
        diff = state_ref[var] - state_test[var]
        n    = diff.size
        errors[var] = {
            'MAE': np.mean(np.abs(diff)),
            'MSE': np.mean(diff**2),
            'L2':  np.sqrt(np.sum(diff**2)) / np.sqrt(np.sum(state_ref[var]**2) + 1e-30),
        }
    return errors


def plot_error_comparison(bubble_amp=2.0, t_end=4.0,
                          ref_dt=0.001, test_dts=None):
    """
    Compute a reference solution at very small dt, then compare
    each scheme at larger dt values. Plot L2 error vs dt.

    ref_dt   : time step for reference solution (very small)
    test_dts : list of dt values to test
    """
    if test_dts is None:
        test_dts = [0.005, 0.01, 0.02, 0.05]

    schemes = ['FTCS', 'BTCS', 'CTCS', 'RK4']
    colors  = ['red', 'orange', 'blue', 'green']

    # --- Compute reference solution ---
    ref_steps = int(t_end / ref_dt)
    print(f"  Computing reference solution: dt={ref_dt}s, "
          f"n_steps={ref_steps}...")
    ref_snaps, grid, _ = run_and_snap('CTCS', bubble_amp,
                                      ref_dt, ref_steps,
                                      save_every=None, silent=True)
    state_ref = ref_snaps[-1][1]
    print(f"  Reference computed at t={ref_snaps[-1][0]:.2f}s")

    # --- Compute errors for each scheme and dt ---
    # errors[scheme][dt][var] = {MAE, MSE, L2}
    all_errors = {s: {} for s in schemes}

    for scheme in schemes:
        for dt in test_dts:
            n_steps = int(t_end / dt)
            print(f"  {scheme}  dt={dt}  steps={n_steps}...", end="")
            snaps, _, blown = run_and_snap(scheme, bubble_amp,
                                           dt, n_steps,
                                           save_every=None, silent=True)
            if blown:
                print(" BLOW-UP — skipped")
                continue

            state_test = snaps[-1][1]
            errs = compute_errors(state_ref, state_test)
            all_errors[scheme][dt] = errs
            print(f" L2(theta')={errs['theta']['L2']:.3e}")

    # --- Plot L2 error vs dt for theta' ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Error vs Δt  |  Reference: CTCS dt={ref_dt}s  |  t_end={t_end}s",
        fontsize=12, fontweight='bold'
    )

    metrics = ['MAE', 'MSE', 'L2']
    var     = 'theta'   # show theta' errors

    for ax, metric in zip(axes, metrics):
        for scheme, color in zip(schemes, colors):
            dts  = sorted(all_errors[scheme].keys())
            vals = [all_errors[scheme][dt][var][metric] for dt in dts]
            if dts:
                ax.loglog(dts, vals, 'o-', color=color,
                          label=scheme, linewidth=1.8, markersize=6)

        # Reference slope lines
        if dts:
            dt_arr = np.array(dts)
            ax.loglog(dt_arr, dt_arr * vals[-1] / dts[-1],
                      'k--', alpha=0.4, linewidth=1, label='1st order')
            ax.loglog(dt_arr, dt_arr**2 * vals[-1] / dts[-1]**2,
                      'k:', alpha=0.4, linewidth=1, label='2nd order')

        ax.set_xlabel("Δt  (s)", fontsize=10)
        ax.set_ylabel(f"{metric}(θ')", fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "error_comparison.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.show()

    return all_errors


# ===========================================================================
# Plot 5: Side-by-side scheme comparison at same dt
# ===========================================================================

def plot_scheme_comparison(bubble_amp=2.0, dt=0.02, n_steps=300):
    """
    Run all stable schemes at the same dt and plot theta' side by side.
    Shows how each scheme handles the warm bubble differently.
    """
    schemes = ['CTCS', 'BTCS', 'RK4']
    fig, axes = plt.subplots(1, len(schemes),
                             figsize=(4.5 * len(schemes), 4.5))

    fig.suptitle(
        f"Scheme comparison — θ' at t={n_steps*dt:.0f}s  |  "
        f"A={bubble_amp}K  dt={dt}s",
        fontsize=12, fontweight='bold'
    )

    all_data = []
    for scheme in schemes:
        print(f"  Running {scheme}...")
        snaps, grid, blown = run_and_snap(scheme, bubble_amp, dt,
                                          n_steps, save_every=None)
        theta = snaps[-1][1]["theta"] if not blown else None
        all_data.append((scheme, theta, blown))

    # Consistent colour scale
    valid = [d for _, d, b in all_data if not b]
    vmax  = max(np.max(np.abs(d)) for d in valid) if valid else 1.0
    x_km  = grid.x_1d / 1000.0
    z_km  = grid.z_1d / 1000.0

    for ax, (scheme, theta, blown) in zip(axes, all_data):
        if blown:
            ax.text(0.5, 0.5, f'{scheme}\nBLOW-UP',
                    ha='center', va='center', fontsize=14,
                    color='red', transform=ax.transAxes)
        else:
            im = ax.pcolormesh(x_km, z_km, theta,
                               cmap='hot_r', vmin=0, vmax=vmax,
                               shading='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='K')

        ax.set_title(scheme, fontsize=11, fontweight='bold')
        ax.set_xlabel("x (km)", fontsize=9)
        ax.set_aspect('equal')

    axes[0].set_ylabel("z (km)", fontsize=9)
    plt.tight_layout()

    fname = os.path.join(PLOT_DIR, "scheme_comparison.png")
    plt.savefig(fname, dpi=130, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.show()


# ===========================================================================
# Main — run all plots
# ===========================================================================

if __name__ == "__main__":

    print("\n=== 1. Warm bubble heatmap (CTCS) ===")
    plot_warm_bubble(scheme='CTCS', bubble_amp=2.0,
                     dt=0.02, n_steps=400, n_frames=5)

    print("\n=== 2. Vertical velocity (CTCS) ===")
    plot_vertical_velocity(scheme='CTCS', bubble_amp=2.0,
                           dt=0.02, n_steps=400, n_frames=5)

    print("\n=== 3. Zero amplitude test — all schemes ===")
    plot_zero_amp_test(dt=0.02, n_steps=100)

    print("\n=== 4. Error comparison ===")
    plot_error_comparison(bubble_amp=2.0, t_end=2.0,
                          ref_dt=0.001,
                          test_dts=[0.005, 0.01, 0.02])

    print("\n=== 5. Scheme comparison ===")
    plot_scheme_comparison(bubble_amp=2.0, dt=0.02, n_steps=300)

    print("\nAll plots saved to output/figures/")
