"""
experiments/robert_bubble.py
============================
Warm bubble benchmark matching Robert (1993) and Pudykiewicz & Clancy (2022).

WHY THIS EXISTS
---------------
warm_bubble.py uses a 1 km domain run for only ~10 s -- the bubble barely moves.
To see the mushroom-cap vortex you need ~25 min on a ~5-10 km domain.
This script uses the correct parameters and runs SI at dt=15 s so the full
integration takes seconds of wall-clock time.

CONFIGURATIONS
--------------
  robert : 10x10 km,  dx=200 m,  dt=15 s,  t_end=900 s  (15 min, 50x50)
  pc     : 5x5 km,    dx=50 m,   dt=15 s,  t_end=1800 s (30 min, 100x100)
  quick  : 5x5 km,    dx=50 m,   dt=15 s,  t_end=900 s  (15 min, 100x100) -- paper quality

USAGE
-----
    python experiments/robert_bubble.py --config quick --scheme SI
    python experiments/robert_bubble.py --config quick --compare
    python experiments/robert_bubble.py --config robert --scheme EPI2
"""

import argparse
import os
import sys
import importlib
import importlib.util
import time as wall_time
sys.path.insert(0, "src")

# Force Python to load src modules from .py source text, bypassing .pyc cache.
# SourceFileLoader checks mtime and may load stale .pyc when the Windows NTFS
# mount does not update file mtimes on Edit. compile()+exec() always uses the
# live .py text on disk regardless of bytecache.
def _load_src(name, path):
    import types
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid        import Grid
from integrators import step, robert_asselin_filter
from results     import save_experiment

PLOT_DIR = "output/figures"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("output/results", exist_ok=True)


# ---------------------------------------------------------------------------
# Pre-set configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    # Robert (1993) domain -- coarse grid (50x50), fast but pixelated
    "robert": {
        "Lx": 10000.0, "Lz": 10000.0, "dx": 200.0, "dz": 200.0,
        "bubble_amp": 2.0,
        "bubble_radius": 2000.0,
        "bubble_xc_frac": 0.5,
        "bubble_zc": 2000.0,
        "t_end": 900.0,
        "dt_si": 15.0,
        "dt_rk4": 1.0,
        "bubble_type": "flattop",
        "label": "Robert (1993) -- 10x10 km, A=2 K, t=15 min",
    },
    # Pudykiewicz & Clancy (2022) Exp. 1 -- finer grid, longer run
    "pc": {
        "Lx": 5000.0, "Lz": 5000.0, "dx": 50.0, "dz": 50.0,
        "bubble_amp": 0.5,
        "bubble_radius": 300.0,
        "bubble_xc_frac": 0.5,
        "bubble_zc": 700.0,
        "t_end": 1800.0,
        "dt_si": 15.0,
        "dt_rk4": 0.1,
        "bubble_type": "gaussian",
        "label": "P&C (2022) Exp.1 -- 5x5 km, dx=50m, A=0.5 K, t=30 min",
    },
    # Quick paper-quality demo -- 100x100 grid, 15 min, clear vortex structure
    "quick": {
        "Lx": 5000.0, "Lz": 5000.0, "dx": 50.0, "dz": 50.0,
        "bubble_amp": 2.0,
        "bubble_radius": 500.0,
        "bubble_xc_frac": 0.5,
        "bubble_zc": 1200.0,
        "t_end": 900.0,
        "dt_si": 15.0,
        "dt_rk4": 0.2,
        "bubble_type": "gaussian",
        "label": "Quick demo -- 5x5 km, dx=50m, A=2 K, t=15 min",
    },
}


# ---------------------------------------------------------------------------
# Initial condition builders
# ---------------------------------------------------------------------------

def _gaussian_bubble(grid, amp, radius, xc, zc):
    """Pure Gaussian theta' perturbation."""
    r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
    return amp * np.exp(-r_sq / radius**2)


def _flattop_bubble(grid, amp, radius, xc, zc, sigma_frac=0.25):
    """Robert (1993) flat-top bubble with cosine-bell transition."""
    sigma = radius * sigma_frac
    r = np.sqrt((grid.x_2d - xc)**2 + (grid.z_2d - zc)**2)
    theta = np.zeros_like(r)
    inside = r <= radius
    transition = (r > radius) & (r <= radius + sigma)
    theta[inside] = amp
    theta[transition] = amp * np.cos(0.5 * np.pi * (r[transition] - radius) / sigma)**2
    return theta


# ---------------------------------------------------------------------------
# Run one scheme
# ---------------------------------------------------------------------------

def run_scheme(cfg, scheme, dt, n_snapshots=7):
    """
    Run the warm bubble with given config and scheme.
    Returns (snapshots, grid, wall_seconds, blown_up).
    """
    grid = Grid({"Lx": cfg["Lx"], "Lz": cfg["Lz"],
                 "dx": cfg["dx"], "dz": cfg["dz"]})

    state     = grid.allocate_state()
    state_old = None

    xc = cfg["Lx"] * cfg["bubble_xc_frac"]
    zc = cfg["bubble_zc"]

    if cfg["bubble_type"] == "flattop":
        state["theta"] = _flattop_bubble(grid, cfg["bubble_amp"],
                                         cfg["bubble_radius"], xc, zc)
    else:
        state["theta"] = _gaussian_bubble(grid, cfg["bubble_amp"],
                                          cfg["bubble_radius"], xc, zc)

    t_end   = cfg["t_end"]
    n_steps = int(round(t_end / dt))

    c_s    = np.sqrt((grid.cp / grid.cv) * grid.Rd * grid.T0)
    cfl_2d = c_s * dt / grid.dx * np.sqrt(2)

    print(f"\n  [{scheme}]  dt={dt}s  steps={n_steps}  "
          f"CFL_2D={cfl_2d:.2f}  T={t_end:.0f}s  "
          f"grid={grid.nx}x{grid.nz}")

    snap_every = max(1, n_steps // (n_snapshots - 1))
    snapshots  = [(0.0, {k: v.copy() for k, v in state.items()})]

    t0_wall = wall_time.perf_counter()
    blown   = False
    t       = 0.0

    for n in range(n_steps):
        state_prev = state_old   # save phi^(n-1) BEFORE step overwrites it
        try:
            state_new, state_old, _ = step(state, grid, dt,
                                           scheme=scheme,
                                           state_old=state_old)
        except Exception as e:
            print(f"  [{scheme}] Exception at step {n+1}: {e}")
            blown = True
            break

        if scheme == 'CTCS' and n > 0:
            state_filtered = robert_asselin_filter(state_prev, state,
                                                   state_new, alpha=0.1)
            state_old = state_filtered
            state     = state_new
        else:
            state = state_new

        t += dt

        if np.any(~np.isfinite(state['u'])):
            print(f"  [{scheme}] BLOW-UP at t={t:.1f}s")
            blown = True
            break

        if (n + 1) % snap_every == 0 or n == n_steps - 1:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

        if (n + 1) % max(1, n_steps // 5) == 0:
            pct = 100 * (n + 1) / n_steps
            print(f"    {pct:5.1f}%  t={t:.1f}s  "
                  f"|theta|={np.max(np.abs(state['theta'])):.3f}K  "
                  f"|w|={np.max(np.abs(state['w'])):.3f}m/s")

    elapsed = wall_time.perf_counter() - t0_wall

    if not blown:
        print(f"  [{scheme}] Done in {elapsed:.1f}s wall time.")
        save_experiment(
            name=f"robert_bubble_{scheme}",
            state=state, grid=grid,
            metadata={"scheme": scheme, "dt": dt, "t_end": t_end,
                      "experiment": "robert_bubble"}
        )

    return snapshots, grid, elapsed, blown


# ---------------------------------------------------------------------------
# Plotting -- paper-quality smooth contourf
# ---------------------------------------------------------------------------

def _panel(ax, x_km, z_km, data, variable, vmin, vmax, title):
    """One panel: filled contours + contour lines (matches Robert 1993 style)."""
    if variable == 'theta':
        cmap       = 'RdYlBu_r'
        n_fill     = 20
        line_col   = 'black'
        neg_col    = None
        cbar_label = "theta' (K)"
    else:
        cmap       = 'RdBu_r'
        n_fill     = 21
        line_col   = 'darkred'
        neg_col    = 'navy'
        cbar_label = "w (m/s)"

    levels_fill = np.linspace(vmin, vmax, n_fill)
    cf = ax.contourf(x_km, z_km, data, levels=levels_fill,
                     cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.92,
                     extend='both')

    n_lines = 7
    pos_levels = np.linspace(0.1 * vmax, vmax, n_lines)
    if np.any(data > 0.05 * vmax):
        cl = ax.contour(x_km, z_km, data, levels=pos_levels,
                        colors=line_col, linewidths=0.8, alpha=0.85)
        ax.clabel(cl, fmt='%.2g', fontsize=6, inline=True)

    if neg_col and np.any(data < -0.05 * abs(vmin)):
        neg_levels = np.linspace(vmin, -0.1 * abs(vmin), n_lines)
        ax.contour(x_km, z_km, data, levels=neg_levels,
                   colors=neg_col, linewidths=0.8,
                   linestyles='dashed', alpha=0.75)

    ax.set_title(title, fontsize=10, fontweight='bold', pad=3)
    ax.set_xlabel("x (km)", fontsize=8)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, lw=0.4)
    return cf, cbar_label


def plot_evolution(snapshots, grid, scheme, variable='theta',
                   cfg_label="", save_suffix=""):
    """Multi-panel time evolution figure (paper style, smooth contours)."""
    snaps = snapshots[::max(1, len(snapshots) // 6)][:6]
    n_panels = len(snaps)

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(3.8 * n_panels, 4.8),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    all_max = max(np.max(np.abs(s[variable])) for _, s in snaps)
    vmax = max(all_max, 1e-6)
    vmin = -vmax if variable == 'w' else 0.0

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0

    last_cf, last_label = None, ''
    for ax, (t, state) in zip(axes, snaps):
        t_min = t / 60.0
        cf, lbl = _panel(ax, x_km, z_km, state[variable],
                         variable, vmin, vmax, f"t = {t_min:.1f} min")
        last_cf, last_label = cf, lbl

    axes[0].set_ylabel("z (km)", fontsize=9)
    fig.colorbar(last_cf, ax=axes, fraction=0.015, pad=0.02,
                 label=last_label)

    var_label = "theta' (K)" if variable == 'theta' else "w (m/s)"
    fig.suptitle(f"Warm bubble -- {var_label}  |  {scheme}\n{cfg_label}",
                 fontsize=11, fontweight='bold')

    fname = os.path.join(PLOT_DIR,
                         f"robert_bubble_{scheme}_{variable}{save_suffix}.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)
    return fname


def plot_comparison(snaps_dict, grid, variable='theta', cfg_label=""):
    """Side-by-side scheme comparison at final time (smooth contours)."""
    schemes = list(snaps_dict.keys())
    n = len(schemes)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    all_max = max(np.max(np.abs(snaps[-1][1][variable]))
                  for snaps in snaps_dict.values())
    vmax = max(all_max, 1e-6)
    vmin = -vmax if variable == 'w' else 0.0

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0

    last_cf, last_label = None, ''
    for ax, scheme in zip(axes, schemes):
        t_final, state = snaps_dict[scheme][-1]
        t_min = t_final / 60.0
        cf, lbl = _panel(ax, x_km, z_km, state[variable],
                         variable, vmin, vmax,
                         f"{scheme}  (t={t_min:.1f} min)")
        last_cf, last_label = cf, lbl

    axes[0].set_ylabel("z (km)", fontsize=9)
    fig.colorbar(last_cf, ax=axes, fraction=0.015, pad=0.02,
                 label=last_label)

    var_label = "theta' (K)" if variable == 'theta' else "w (m/s)"
    fig.suptitle(f"Scheme comparison -- {var_label}\n{cfg_label}",
                 fontsize=11, fontweight='bold')

    fname = os.path.join(PLOT_DIR,
                         f"robert_bubble_comparison_{variable}.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robert (1993) warm bubble -- correct domain and time scale"
    )
    parser.add_argument("--config",  default="quick",
                        choices=list(CONFIGS.keys()),
                        help="Parameter set (default: quick)")
    parser.add_argument("--scheme",  default="SI",
                        choices=['FTCS','CTCS','RK4','SI','EPI2','EPI3'],
                        help="Time scheme (default: SI)")
    parser.add_argument("--dt",      type=float, default=None,
                        help="Override time step (default: from config)")
    parser.add_argument("--compare", action="store_true",
                        help="Run SI and RK4 side-by-side for comparison")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    print("\n" + "="*60)
    print(f"  Config : {args.config}")
    print(f"  {cfg['label']}")
    print("="*60)

    if args.compare:
        snaps_dict = {}
        for scheme, dt_key in [("SI", "dt_si"), ("RK4", "dt_rk4")]:
            dt = args.dt if args.dt else cfg[dt_key]
            snaps, grid, elapsed, blown = run_scheme(cfg, scheme, dt)
            if not blown:
                snaps_dict[scheme] = snaps
                plot_evolution(snaps, grid, scheme, 'theta',
                               cfg['label'], f"_{args.config}")
                plot_evolution(snaps, grid, scheme, 'w',
                               cfg['label'], f"_{args.config}")

        if len(snaps_dict) == 2:
            plot_comparison(snaps_dict, grid, 'theta', cfg_label=cfg['label'])
            plot_comparison(snaps_dict, grid, 'w',     cfg_label=cfg['label'])

    else:
        dt_key = "dt_rk4" if args.scheme.upper() == "RK4" else "dt_si"
        dt = args.dt if args.dt else cfg[dt_key]
        snaps, grid, elapsed, blown = run_scheme(cfg, scheme=args.scheme, dt=dt)
        if not blown:
            plot_evolution(snaps, grid, args.scheme, 'theta',
                           cfg['label'], f"_{args.config}")
            plot_evolution(snaps, grid, args.scheme, 'w',
                           cfg['label'], f"_{args.config}")

    print("\nAll done. Figures in output/figures/")
