"""
experiments/gr_bubble.py
========================
Rising Thermal Bubble benchmark — Giraldo & Restelli (2008) Case 2, Section 3.2.

Recommended by Dr Colm Clancy (email, June 2026) as the test case that runs
cleanly without explicit diffusion or filtering, unlike the P&C (2022) bubble
which requires diffusion to avoid crashing before t=30 min.

Case 2 parameters (G&R 2008, Section 3.2)
------------------------------------------
  Domain    : (x,z) ∈ [0, 1000] × [0, 1000] m
  θ̄         : 300 K (constant, neutral stratification)
  Bubble    : θ' = (θc/2)(1 + cos(π·r/rc))  for r ≤ rc, else 0
              θc = 0.5 K,  rc = 250 m,  centre (xc,zc) = (500, 350) m
  t_end     : 700 s
  BCs       : no-flux all four walls (we use periodic-x as the bubble is
              centred and doesn't reach the walls in 700 s)

Reference: Giraldo, F.X. and Restelli, M. (2008).
  "A study of spectral element and discontinuous Galerkin methods for the
   Navier–Stokes equations in nonhydrostatic mesoscale atmospheric modeling:
   Equation sets and test cases."  J. Comput. Phys. 227, 3849–3877.
   https://doi.org/10.1016/j.jcp.2007.12.009

Usage
-----
    python experiments/gr_bubble.py --scheme RK4
    python experiments/gr_bubble.py --scheme SI
    python experiments/gr_bubble.py --scheme EPI2
    python experiments/gr_bubble.py --scheme EPI3
    python experiments/gr_bubble.py --compare          # RK4 vs SI vs EPI2 vs EPI3
    python experiments/gr_bubble.py --scheme RK4 --dx 20   # coarser grid for speed
"""

import argparse
import os
import sys
import time as wall_time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def _load_src(name, path):
    """Force live .py load (bypasses stale .pyc on Windows NTFS mount)."""
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    with open(path, 'r') as f:
        source = f.read()
    exec(compile(source, os.path.abspath(path), 'exec'), mod.__dict__)
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
# G&R (2008) Case 2 configuration
# ---------------------------------------------------------------------------

GR_CONFIG = {
    # Domain — same 1 km × 1 km as the existing small-domain tests
    "Lx": 1000.0,   # m
    "Lz": 1000.0,   # m
    "dx":   10.0,   # m   -> nx = nz = 100 (fast, matches existing tests)
    "dz":   10.0,   # m

    # Warm cosine bubble  (G&R 2008 eq. in Section 3.2)
    "bubble_theta_c": 0.5,    # K    amplitude
    "bubble_rc":     250.0,   # m    radius
    "bubble_xc":     500.0,   # m    x centre
    "bubble_zc":     350.0,   # m    z centre (G&R: lower than domain mid)

    # Time steps
    #   RK4/CTCS: explicit acoustic CFL limit ≈ dx/(c_s√2) ≈ 0.020 s at dx=10 m
    #   SI / EPI : no acoustic CFL → large dt fine (Courant ~ 520 at dt=15 s, dx=10 m)
    "dt_RK4":  0.01,   # s   (safe explicit; 70 000 steps for 700 s)
    "dt_CTCS": 0.01,   # s
    "dt_SI":   15.0,   # s   (47 steps for 700 s)
    "dt_EPI2": 15.0,   # s
    "dt_EPI3": 15.0,   # s

    # Output: save snapshot at these times (s)
    "save_times": [0, 100, 200, 300, 400, 500, 600, 700],  # s
    "t_end":      700.0,   # s
}

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
    G&R (2008) Section 3.2 — cosine warm bubble:

        θ'(x,z) = (θc/2)(1 + cos(π·r/rc))   for r ≤ rc
                  0                            for r > rc

    where r = √[(x-xc)² + (z-zc)²].
    u, w, π' initialised to zero; atmosphere at rest and in hydrostatic balance.
    """
    state  = grid.allocate_state()
    xc     = cfg["bubble_xc"]
    zc     = cfg["bubble_zc"]
    theta_c = cfg["bubble_theta_c"]
    rc      = cfg["bubble_rc"]

    r = np.sqrt((grid.x_2d - xc)**2 + (grid.z_2d - zc)**2)
    state["theta"] = np.where(
        r <= rc,
        0.5 * theta_c * (1.0 + np.cos(np.pi * r / rc)),
        0.0
    )
    return state


# ---------------------------------------------------------------------------
# Run one scheme
# ---------------------------------------------------------------------------

def run_scheme(scheme, cfg, dt, t_end, use_shapiro=False, verbose=True):
    """
    Run G&R Case 2 from t=0 to t_end with the chosen scheme.

    Parameters
    ----------
    scheme      : str   — 'RK4', 'SI', 'EPI2', 'EPI3', etc.
    cfg         : dict  — GR_CONFIG (or overridden version)
    dt          : float — time step (s)
    t_end       : float — end time (s)
    use_shapiro : bool  — apply Shapiro filter every 2 EPI steps (default off;
                          G&R Case 2 is stable without filtering)

    Returns
    -------
    snapshots   : list of (t_s, state_dict)
    grid        : Grid
    elapsed_s   : float   wall-clock seconds
    blown_up    : bool
    """
    grid      = Grid({"Lx": cfg["Lx"], "Lz": cfg["Lz"],
                      "dx": cfg["dx"], "dz": cfg["dz"]})
    state     = make_initial_state(grid, cfg)
    state_old  = None
    epi_n_prev = None
    is_epi     = scheme in ('EPI2', 'EPI3')

    n_steps = int(round(t_end / dt))
    save_times = set(cfg.get("save_times", []))
    save_every  = max(1, int(round(100.0 / dt)))   # fallback: every 100 s

    c_s = np.sqrt((grid.cp / grid.cv) * grid.Rd * grid.T0)
    cfl = c_s * dt / grid.dx

    if verbose:
        print(f"\n  [{scheme}]  grid={grid.nx}×{grid.nz}  "
              f"dx={grid.dx:.0f}m  dt={dt}s  "
              f"acoustic CFL={cfl:.1f}  "
              f"steps={n_steps}  T={t_end:.0f}s")
        if cfl > 1.0 and scheme in ('FTCS', 'BTCS', 'CTCS', 'RK4'):
            print(f"  WARNING: CFL={cfl:.2f} > 1 for explicit scheme")

    # Include t=0 snapshot
    snapshots = [(0.0, {k: v.copy() for k, v in state.items()})]
    t0_wall   = wall_time.perf_counter()
    t         = 0.0
    blown     = False

    for n in range(n_steps):
        state_old_saved = state_old

        try:
            state_new, state_old, epi_extra = step(
                state, grid, dt, scheme=scheme,
                state_old=state_old, epi_n_prev=epi_n_prev
            )
        except Exception as e:
            print(f"  [{scheme}] Exception at step {n+1} (t={t:.1f}s): {e}")
            blown = True
            break

        if scheme == 'CTCS' and n > 0:
            state = robert_asselin_filter(state_old_saved, state,
                                          state_new, alpha=0.1)
        else:
            state = state_new

        if is_epi and epi_extra is not None:
            epi_n_prev = epi_extra['n_rhs']

        # Optional Shapiro filter (off by default for G&R Case 2)
        if use_shapiro and is_epi and (n + 1) % 2 == 0:
            state = shapiro_filter(state, grid)

        t += dt

        if not np.all(np.isfinite(state['u'])):
            print(f"  [{scheme}] BLOW-UP at t={t:.1f}s")
            blown = True
            break

        # Save at requested times (within one dt tolerance)
        for ts in save_times:
            if abs(t - ts) < 0.5 * dt and ts > 0:
                snapshots.append((t, {k: v.copy() for k, v in state.items()}))

        if verbose and (n + 1) % max(1, int(round(100.0 / dt))) == 0:
            print(f"  [{scheme}]  t={t:6.1f}s  "
                  f"|θ'|={np.max(np.abs(state['theta'])):.4f} K  "
                  f"|w|={np.max(np.abs(state['w'])):.4f} m/s")

    elapsed = wall_time.perf_counter() - t0_wall

    if not blown:
        # Ensure final state saved
        if not snapshots or abs(snapshots[-1][0] - t) > 0.5 * dt:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

        if verbose:
            print(f"\n  [{scheme}] Done: {elapsed:.1f}s wall time  "
                  f"({len(snapshots)} snapshots saved)")

        exp_name = (f"gr_bubble_{scheme}_"
                    f"dt{str(dt).replace('.', 'p')}_"
                    f"dx{int(cfg['dx'])}m_T{int(t_end)}s")
        save_experiment(
            name      = exp_name,
            state     = state,
            grid      = grid,
            snapshots = snapshots,
            metadata  = {
                "experiment":  "gr_bubble_2008_case2",
                "scheme":      scheme,
                "dt":          dt,
                "t_end":       t_end,
                "bubble_theta_c": cfg["bubble_theta_c"],
                "bubble_rc":      cfg["bubble_rc"],
                "bubble_zc":      cfg["bubble_zc"],
                "dx":          cfg["dx"],
                "nx":          grid.nx,
                "nz":          grid.nz,
                "wall_time_s": elapsed,
            }
        )

    return snapshots, grid, elapsed, blown


# ---------------------------------------------------------------------------
# Plot: θ' contour panels at t=0 and t=700 s  (G&R Fig. 3 style)
# ---------------------------------------------------------------------------

def plot_result(snapshots, grid, scheme, cfg, filename=None):
    """Reproduce G&R Figure 3: θ' contours at final time (and optionally t=0)."""
    if len(snapshots) < 1:
        print("  No snapshots to plot.")
        return

    # Use last snapshot
    t_final, state_final = snapshots[-1]

    X = grid.x_2d / 1000.0   # km
    Z = grid.z_2d / 1000.0   # km

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        f"G&R (2008) Case 2 — Rising Thermal Bubble — {scheme}  "
        f"(dx={grid.dx:.0f}m, dt={cfg[_DT_KEY.get(scheme,'dt_RK4')]}s)",
        fontsize=11
    )

    # Panel 1: t=0
    t0, st0 = snapshots[0]
    levels0 = np.linspace(0, 0.5, 11)
    cs0 = axes[0].contourf(X, Z, st0['theta'], levels=levels0,
                            cmap='RdYlBu_r', extend='neither')
    axes[0].contour(X, Z, st0['theta'], levels=levels0, colors='k',
                    linewidths=0.5, linestyles='-')
    plt.colorbar(cs0, ax=axes[0], label="θ' (K)")
    axes[0].set_title(f"t = {t0:.0f} s (initial)")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("z (km)")
    axes[0].set_aspect('equal')

    # Panel 2: final time
    vmax = max(0.01, np.max(np.abs(state_final['theta'])))
    levels_f = np.linspace(-0.05, vmax, 13)
    cs1 = axes[1].contourf(X, Z, state_final['theta'], levels=levels_f,
                            cmap='RdYlBu_r', extend='both')
    axes[1].contour(X, Z, state_final['theta'], levels=levels_f, colors='k',
                    linewidths=0.5, linestyles='-')
    plt.colorbar(cs1, ax=axes[1], label="θ' (K)")
    axes[1].set_title(f"t = {t_final:.0f} s")
    axes[1].set_xlabel("x (km)")
    axes[1].set_ylabel("z (km)")
    axes[1].set_aspect('equal')

    plt.tight_layout()

    if filename is None:
        filename = os.path.join(PLOT_DIR,
                                f"gr_bubble_{scheme}_dx{int(grid.dx)}m.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {filename}")
    return filename


# ---------------------------------------------------------------------------
# Compare all schemes (efficiency table + side-by-side θ' at t=700 s)
# ---------------------------------------------------------------------------

def run_compare(cfg, schemes=('RK4', 'SI', 'EPI2', 'EPI3'), t_end=None):
    """Run all schemes and print a wall-clock comparison table."""
    if t_end is None:
        t_end = cfg["t_end"]

    results = {}
    for scheme in schemes:
        dt = cfg[_DT_KEY.get(scheme, 'dt_RK4')]
        snaps, grid, elapsed, blown = run_scheme(scheme, cfg, dt, t_end)
        results[scheme] = {
            "snaps": snaps, "grid": grid, "elapsed": elapsed, "blown": blown,
            "dt": dt,
            "n_steps": int(round(t_end / dt)),
        }
        if not blown:
            plot_result(snaps, grid, scheme, cfg)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  G&R Case 2  t_end={t_end:.0f}s  dx={cfg['dx']:.0f}m")
    print(f"  {'Scheme':6s}  {'dt':8s}  {'steps':7s}  "
          f"{'wall(s)':9s}  {'|w|_max':9s}  {'status':8s}")
    print("-" * 60)
    for scheme, r in results.items():
        if not r["blown"] and r["snaps"]:
            _, sf = r["snaps"][-1]
            wmax = np.max(np.abs(sf['w']))
            print(f"  {scheme:6s}  {r['dt']:8.3f}  {r['n_steps']:7d}  "
                  f"{r['elapsed']:9.1f}  {wmax:9.4f}  {'OK':8s}")
        else:
            print(f"  {scheme:6s}  {r['dt']:8.3f}  {r['n_steps']:7d}  "
                  f"  {'---':9s}  {'---':9s}  {'BLOWN':8s}")
    print("=" * 60)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G&R (2008) Case 2 rising thermal bubble benchmark"
    )
    parser.add_argument('--scheme',  default='RK4',
                        choices=['FTCS','BTCS','CTCS','RK4','SI','EPI2','EPI3'],
                        help='Time integration scheme (default: RK4)')
    parser.add_argument('--t_end',  type=float, default=None,
                        help='End time in seconds (default: 700)')
    parser.add_argument('--dx',     type=float, default=None,
                        help='Grid spacing in m (default: 10)')
    parser.add_argument('--dt',     type=float, default=None,
                        help='Override time step (s)')
    parser.add_argument('--compare', action='store_true',
                        help='Run all four schemes and print comparison table')
    parser.add_argument('--shapiro', action='store_true',
                        help='Apply Shapiro filter every 2 EPI steps (off by default)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip figure output')
    args = parser.parse_args()

    cfg = dict(GR_CONFIG)  # shallow copy so we can override
    if args.dx is not None:
        cfg["dx"] = args.dx
        cfg["dz"] = args.dx
    t_end = args.t_end if args.t_end is not None else cfg["t_end"]

    if args.compare:
        run_compare(cfg, t_end=t_end)
        return

    scheme = args.scheme
    dt     = args.dt if args.dt is not None else cfg[_DT_KEY.get(scheme, 'dt_RK4')]

    snaps, grid, elapsed, blown = run_scheme(
        scheme, cfg, dt, t_end,
        use_shapiro=args.shapiro,
    )

    if not blown and not args.no_plot:
        plot_result(snaps, grid, scheme, cfg)

    # Quick summary
    if not blown and snaps:
        _, sf = snaps[-1]
        print(f"\n  Final state (t={t_end:.0f}s):  "
              f"|θ'|_max = {np.max(np.abs(sf['theta'])):.4f} K  "
              f"|w|_max = {np.max(np.abs(sf['w'])):.4f} m/s")


if __name__ == "__main__":
    main()
