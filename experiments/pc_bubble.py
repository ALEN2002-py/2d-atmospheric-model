"""
experiments/pc_bubble.py
========================
Warm bubble benchmark from Pudykiewicz & Clancy (2022) Experiment 1.

Dr Clancy's exact instructions (meeting, June 2026):
  - 5 km x 5 km domain
  - dx = dz = 20 m  ->  250 x 250 grid points
  - Gaussian warm bubble, centre at z ~ 700 m, A = 2 K, r = 500 m
  - Run for 10-30 minutes; things start happening around 10-15 min
  - Save snapshot every 5 minutes
  - SI at dt=15s (explicit needs dt < dx/c_s ~ 0.058s -- 250x slower)
  - Compare SI vs RK4 side by side

Paper reference values (P&C 2022 Exp. 1):
  Domain : 5x5 km,  dx=20 m,  250x250
  Bubble : theta'=0.5 K cylinder (we use Gaussian, A=2K per Clancy)
  dt_SI  : 15 s,  Courant ~ 250
  t_end  : 30 min

Usage
-----
    # 15-min SI (quick, ~30s wall-clock)
    python experiments/pc_bubble.py --scheme SI --t_end 900

    # 30-min SI vs RK4 comparison (the paper result)
    python experiments/pc_bubble.py --compare --t_end 1800

    # Quick 5-min check
    python experiments/pc_bubble.py --scheme SI --t_end 300

    # Coarser grid for fast testing
    python experiments/pc_bubble.py --scheme SI --t_end 900 --dx 50
"""

import argparse
import os
import sys
import time as wall_time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Force Python to load src modules from .py source, bypassing stale .pyc cache.
# Windows NTFS mount does not update file mtimes on Edit, so Python's
# SourceFileLoader may load old compiled bytecache. compile()+exec() always
# reads the live .py text regardless of .pyc.
def _load_src(name, path):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    with open(path, 'r') as f:
        source = f.read()
    code = compile(source, os.path.abspath(path), 'exec')
    exec(code, mod.__dict__)
    return mod

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_load_src("grid",        os.path.join(_src, "grid.py"))
_load_src("dynamics",    os.path.join(_src, "dynamics.py"))
_load_src("integrators", os.path.join(_src, "integrators.py"))
_load_src("results",     os.path.join(_src, "results.py"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid        import Grid
from integrators import step, robert_asselin_filter, shapiro_filter
from results     import save_experiment

PLOT_DIR    = "output/figures"
RESULTS_DIR = "output/results"
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# P&C (2022) configuration  --  Dr Clancy's numbers from June 2026 meeting
# ---------------------------------------------------------------------------

PC_CONFIG = {
    # Domain
    "Lx": 5000.0,   # m   (5 km)
    "Lz": 5000.0,   # m   (5 km)
    "dx":   20.0,   # m   -> nx = 250
    "dz":   20.0,   # m   -> nz = 250

    # Warm bubble initial condition  --  P&C (2022) eq. 6.1
    # Cylindrical flat-top with Gaussian edge smoothing
    "bubble_amp":   0.5,    # K    AT in eq. 6.1
    "bubble_a":     400.0,  # m    flat-top radius  (a = 20*delta = 20*20)
    "bubble_sigma": 100.0,  # m    edge smoothing   (sigma = 5*delta = 5*20)
    "bubble_xc":    2500.0, # m    x0 = Lx/2
    "bubble_zc":    700.0,  # m    z0 = 700 m (as per paper)

    # Time steps
    "dt_SI":   15.0,   # s   SI removes acoustic CFL -- large dt fine
    "dt_RK4":  0.03,   # s   RK4: 2D CFL limit = dx/(c_s*sqrt(2)) = 0.041s; 0.03 is safe
    "dt_CTCS": 0.03,   # s   same as RK4
    "dt_EPI2": 15.0,   # s   EPI also removes acoustic CFL
    "dt_EPI3": 15.0,   # s

    # Output: save snapshot every 5 minutes (Clancy: "5 min, 10, 15, 20, 30")
    "save_interval": 300.0,  # s
}

# dt lookup by scheme key
_DT_KEY = {
    'FTCS': 'dt_RK4', 'BTCS': 'dt_RK4', 'CTCS': 'dt_CTCS',
    'RK4':  'dt_RK4',
    'SI':   'dt_SI',
    'EPI2': 'dt_EPI2', 'EPI3': 'dt_EPI3',
}


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def make_initial_state(grid, cfg):
    """
    P&C (2022) eq. 6.1 -- cylindrical flat-top with Gaussian edge smoothing.

        theta' = AT                                   for r <= a
        theta' = AT * exp(-(r-a)^2 / (2*sigma^2))   for r >  a

    with AT=0.5K, a=400m (20*delta), sigma=100m (5*delta), z0=700m.
    u, w, pi' start at zero.
    """
    state = grid.allocate_state()
    xc    = cfg["bubble_xc"]
    zc    = cfg["bubble_zc"]
    AT    = cfg["bubble_amp"]
    a     = cfg["bubble_a"]
    sigma = cfg["bubble_sigma"]

    r = np.sqrt((grid.x_2d - xc)**2 + (grid.z_2d - zc)**2)

    theta = np.where(
        r <= a,
        AT,                                          # flat top inside radius a
        AT * np.exp(-((r - a)**2) / (2.0 * sigma**2))  # Gaussian tail outside
    )
    state["theta"] = theta
    return state


# ---------------------------------------------------------------------------
# Run one scheme
# ---------------------------------------------------------------------------

def run_scheme(scheme, cfg, dt, t_end, verbose=True):
    """
    Run warm bubble from t=0 to t_end with the given scheme.

    Saves a snapshot at t=0 and then every cfg['save_interval'] seconds.
    Returns (snapshots, grid, wall_time_s, blown_up).

    snapshots : list of (t_seconds, state_dict)
    """
    grid      = Grid({"Lx": cfg["Lx"], "Lz": cfg["Lz"],
                      "dx": cfg["dx"], "dz": cfg["dz"]})
    state     = make_initial_state(grid, cfg)
    state_old = None   # CTCS bootstraps with FTCS on step 0
    epi_n_prev = None  # EPI3 multi-step: stores N^{n-1}
    is_epi    = scheme in ('EPI2', 'EPI3')

    n_steps    = int(round(t_end / dt))
    save_every = max(1, int(round(cfg["save_interval"] / dt)))

    # Acoustic CFL (informational -- SI/EPI ignore it)
    c_s = np.sqrt((grid.cp / grid.cv) * grid.Rd * grid.T0)
    cfl = c_s * dt / grid.dx

    if verbose:
        print(f"\n  [{scheme}]  grid={grid.nx}x{grid.nz}  "
              f"dx={grid.dx:.0f}m  dt={dt}s  "
              f"acoustic CFL={cfl:.1f}  "
              f"steps={n_steps}  T={t_end/60:.0f} min")
        if cfl > 1.0 and scheme in ('FTCS', 'BTCS', 'CTCS', 'RK4'):
            print(f"  WARNING: CFL={cfl:.2f} > 1 for explicit scheme  "
                  f"(stable max dt ~ {grid.dx/c_s:.3f}s)")

    # t=0 snapshot
    snapshots = [(0.0, {k: v.copy() for k, v in state.items()})]
    t0_wall   = wall_time.perf_counter()
    t         = 0.0
    blown     = False

    for n in range(n_steps):
        # Save phi^(n-1) BEFORE step() overwrites state_old
        state_old_saved = state_old

        try:
            state_new, state_old, epi_extra = step(state, grid, dt,
                                                    scheme=scheme,
                                                    state_old=state_old,
                                                    epi_n_prev=epi_n_prev)
        except Exception as e:
            print(f"  [{scheme}] Exception at step {n+1} (t={t:.1f}s): {e}")
            blown = True
            break

        # Robert-Asselin filter for leapfrog
        if scheme == 'CTCS' and n > 0:
            state = robert_asselin_filter(state_old_saved, state,
                                          state_new, alpha=0.1)
        else:
            state = state_new

        # EPI: track N^{n-1} for EPI3 multi-step correction
        if is_epi and epi_extra is not None:
            epi_n_prev = epi_extra['n_rhs']

        # Shapiro filter every 2 EPI steps (P&C 2022 eq 5.6-5.7)
        if is_epi and (n + 1) % 2 == 0:
            state = shapiro_filter(state, grid)

        t += dt

        # Blow-up check
        if not np.all(np.isfinite(state['u'])):
            print(f"  [{scheme}] BLOW-UP at t={t:.1f}s ({t/60:.1f} min)")
            blown = True
            break

        # Save snapshot every 5 minutes
        if (n + 1) % save_every == 0:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))
            if verbose:
                print(f"  [{scheme}]  t={t/60:5.1f} min  "
                      f"|theta'|={np.max(np.abs(state['theta'])):.3f} K  "
                      f"|w|={np.max(np.abs(state['w'])):.4f} m/s")

    elapsed = wall_time.perf_counter() - t0_wall

    if not blown:
        # Ensure the final state is included even if t_end is not a multiple
        if snapshots[-1][0] < t - 0.5 * dt:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

        if verbose:
            print(f"\n  [{scheme}] Complete: {elapsed:.1f}s wall time  "
                  f"({len(snapshots)} snapshots)")

        # Persist to disk
        exp_name = (f"pc_bubble_{scheme}_"
                    f"dt{str(dt).replace('.', 'p')}_"
                    f"T{int(t_end)}s")
        save_experiment(
            name      = exp_name,
            state     = state,
            grid      = grid,
            snapshots = snapshots,
            metadata  = {
                "experiment":  "pc_bubble_2022",
                "scheme":      scheme,
                "dt":          dt,
                "t_end":       t_end,
                "bubble_amp":  cfg["bubble_amp"],
                "bubble_zc":   cfg["bubble_zc"],
                "dx":          cfg["dx"],
                "nx":          grid.nx,
                "nz":          grid.nz,
                "wall_time_s": elapsed,
            }
        )

    return snapshots, grid, elapsed, blown


# ---------------------------------------------------------------------------
# Publication-quality contour plots  (Clancy / P&C 2022 style)
# ---------------------------------------------------------------------------

def _draw_panel(ax, X, Z, data, variable, vmin, vmax, title, grid_Lz_km):
    """Filled contours + labelled contour lines on one axes panel."""
    if variable == 'theta':
        cmap       = 'RdYlBu_r'
        lv_fill    = np.linspace(0,    vmax,  20)
        lv_pos     = np.linspace(0.1 * vmax, vmax, 8)
        lv_neg     = None
        cbar_label = "theta' (K)"
        line_col   = 'black'
    else:
        cmap       = 'RdBu_r'
        lv_fill    = np.linspace(-vmax, vmax, 21)
        lv_pos     = np.linspace( 0.05 * vmax,  vmax, 7)
        lv_neg     = np.linspace(-vmax, -0.05 * vmax, 7)
        cbar_label = "w (m/s)"
        line_col   = 'darkred'

    cf = ax.contourf(X, Z, data, levels=lv_fill, cmap=cmap,
                     vmin=vmin, vmax=vmax, alpha=0.90, extend='both')

    if np.any(data > 0.05 * vmax):
        cl = ax.contour(X, Z, data, levels=lv_pos,
                        colors=line_col, linewidths=0.8, alpha=0.85)
        ax.clabel(cl, fmt='%.2g', fontsize=6, inline=True, inline_spacing=2)

    if lv_neg is not None and np.any(data < -0.05 * abs(vmin)):
        ax.contour(X, Z, data, levels=lv_neg,
                   colors='navy', linewidths=0.8,
                   linestyles='dashed', alpha=0.75)

    # Sponge layer boundary (top 20% of domain)
    ax.axhline(grid_Lz_km * 0.8, color='steelblue',
               lw=0.8, linestyle='--', alpha=0.5, label='sponge')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
    ax.set_xlabel("x  (km)", fontsize=9)
    ax.set_xlim(0, X.max())
    ax.set_ylim(0, Z.max())
    ax.set_aspect('equal')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.15, lw=0.4)
    return cf, cbar_label


def plot_evolution(snapshots, grid, scheme, variable, cfg, t_end,
                   n_panels=6, save_suffix=""):
    """
    Time-evolution figure: one column per snapshot.
    Matches Robert (1993) / P&C (2022) multi-panel style.
    """
    step_sz = max(1, (len(snapshots) - 1) // (n_panels - 1))
    snaps   = snapshots[::step_sz][:n_panels]

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0
    X, Z = np.meshgrid(x_km, z_km)

    n_snaps = len(snaps)
    n_cols  = 2
    n_rows  = (n_snaps + 1) // 2   # ceil division -> 2 rows for 4 panels
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5.0 * n_rows),
                             constrained_layout=True)
    axes_flat = np.array(axes).ravel()

    # Consistent colour scale across all panels
    all_vals = np.concatenate([s[variable].ravel() for _, s in snaps])
    vmax = max(float(np.max(np.abs(all_vals))), 1e-10)
    vmin = 0.0 if variable == 'theta' else -vmax

    last_cf, last_label = None, ''
    for ax, (t, state) in zip(axes_flat, snaps):
        cf, lbl = _draw_panel(ax, X, Z, state[variable],
                              variable, vmin, vmax,
                              f"t = {t/60:.0f} min",
                              grid.Lz / 1000.0)
        last_cf, last_label = cf, lbl

    # Hide any unused axes (e.g. if odd number of snapshots)
    for ax in axes_flat[n_snaps:]:
        ax.set_visible(False)

    for row_axes in np.array(axes).reshape(n_rows, n_cols):
        row_axes[0].set_ylabel("z  (km)", fontsize=10)
    fig.colorbar(last_cf, ax=axes_flat[:n_snaps], shrink=0.6, pad=0.02, label=last_label)

    var_lbl = "theta'" if variable == 'theta' else "w"
    fig.suptitle(
        f"P&C (2022) Warm Bubble — {var_lbl}\n"
        f"Scheme: {scheme}  |  A={cfg['bubble_amp']} K  |  "
        f"dx=dz={cfg['dx']} m  |  T={t_end/60:.0f} min",
        fontsize=12, fontweight='bold'
    )

    fname = os.path.join(PLOT_DIR,
                         f"pc_bubble_{scheme}_{variable}"
                         f"_T{int(t_end)}s{save_suffix}.png")
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {fname}")
    return fname


def plot_scheme_comparison(snaps_dict, grid, variable, cfg, t_end):
    """
    Side-by-side final-time comparison across schemes.
    The key dissertation figure.
    """
    schemes = list(snaps_dict.keys())
    n = len(schemes)

    fig, axes = plt.subplots(1, n,
                             figsize=(5.5 * n, 5.5),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([
        snaps[-1][1][variable].ravel()
        for snaps in snaps_dict.values()
    ])
    vmax = max(float(np.max(np.abs(all_vals))), 1e-10)
    vmin = 0.0 if variable == 'theta' else -vmax

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0
    X, Z = np.meshgrid(x_km, z_km)

    last_cf, last_label = None, ''
    for ax, scheme in zip(axes, schemes):
        t_final, state = snaps_dict[scheme][-1]
        cf, lbl = _draw_panel(ax, X, Z, state[variable],
                              variable, vmin, vmax,
                              f"{scheme}  (t={t_final/60:.0f} min)",
                              grid.Lz / 1000.0)
        last_cf, last_label = cf, lbl

    axes[0].set_ylabel("z  (km)", fontsize=10)
    fig.colorbar(last_cf, ax=axes, shrink=0.8, pad=0.02, label=last_label)

    var_lbl = "theta'" if variable == 'theta' else "w"
    fig.suptitle(
        f"P&C (2022) Scheme Comparison — {var_lbl}\n"
        f"A={cfg['bubble_amp']} K  |  dx=dz={cfg['dx']} m  |  "
        f"T={t_end/60:.0f} min",
        fontsize=12, fontweight='bold'
    )

    fname = os.path.join(PLOT_DIR,
                         f"pc_bubble_comparison_{variable}_T{int(t_end)}s.png")
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparison plot saved: {fname}")
    return fname


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P&C (2022) warm bubble  --  5x5 km, 250x250 grid"
    )
    parser.add_argument("--scheme", default="SI",
                        choices=['FTCS', 'CTCS', 'RK4', 'SI', 'EPI2', 'EPI3'],
                        help="Time scheme (default: SI)")
    parser.add_argument("--t_end", type=float, default=900.0,
                        help="End time in seconds (default: 900 = 15 min)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override time step (s)")
    parser.add_argument("--dx", type=float, default=None,
                        help="Override grid spacing in m (default: 20 m = 250x250)")
    parser.add_argument("--compare", action="store_true",
                        help="Run SI + RK4 side-by-side and plot comparison")
    args = parser.parse_args()

    cfg = dict(PC_CONFIG)
    if args.dx:
        cfg["dx"] = cfg["dz"] = float(args.dx)

    nx = int(cfg["Lx"] / cfg["dx"])
    nz = int(cfg["Lz"] / cfg["dz"])

    print("\n" + "=" * 62)
    print("  Pudykiewicz & Clancy (2022)  Warm Bubble  Experiment 1")
    print(f"  Domain  : {cfg['Lx']/1000:.0f} x {cfg['Lz']/1000:.0f} km")
    print(f"  Grid    : dx=dz={cfg['dx']:.0f} m  ({nx} x {nz} points)")
    print(f"  Bubble  : A={cfg['bubble_amp']} K  "
          f"a={cfg['bubble_a']} m  sigma={cfg['bubble_sigma']} m  zc={cfg['bubble_zc']} m")
    print(f"  T_end   : {args.t_end/60:.0f} min  "
          f"(snapshots every {cfg['save_interval']/60:.0f} min)")
    print("=" * 62)

    if args.compare:
        snaps_dict = {}
        grid_ref   = None

        for scheme in ["SI", "RK4"]:
            dt = args.dt if args.dt else cfg[_DT_KEY[scheme]]
            snaps, grid_ref, elapsed, blown = run_scheme(
                scheme, cfg, dt, args.t_end
            )
            if not blown:
                snaps_dict[scheme] = snaps
                plot_evolution(snaps, grid_ref, scheme, 'theta', cfg, args.t_end)
                plot_evolution(snaps, grid_ref, scheme, 'w',     cfg, args.t_end)

        if len(snaps_dict) == 2:
            plot_scheme_comparison(snaps_dict, grid_ref, 'theta', cfg, args.t_end)
            plot_scheme_comparison(snaps_dict, grid_ref, 'w',     cfg, args.t_end)
        elif len(snaps_dict) == 1:
            print("\n  One scheme blew up -- no comparison plot generated.")

    else:
        dt = args.dt if args.dt else cfg[_DT_KEY[args.scheme]]
        snaps, grid, elapsed, blown = run_scheme(
            args.scheme, cfg, dt, args.t_end
        )
        if not blown:
            plot_evolution(snaps, grid, args.scheme, 'theta', cfg, args.t_end)
            plot_evolution(snaps, grid, args.scheme, 'w',     cfg, args.t_end)

    print("\n  All outputs in output/figures/  and  output/results/")
