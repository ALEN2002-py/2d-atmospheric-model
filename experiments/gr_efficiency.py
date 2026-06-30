"""
experiments/gr_efficiency.py
=============================
Efficiency study for G&R (2008) Case 2 rising thermal bubble.

For each scheme, runs at multiple dt values and measures:
  - L2 relative error vs reference (RK4 at dt_ref)
  - Total wall-clock time

Produces the key dissertation figure: error vs wall time (log-log).

Reference solution
------------------
  RK4  dt=0.005 s  (140 000 steps, ~8 min on modern hardware)
  Saved to disk on first run; reloaded on subsequent runs.

Test dt values
--------------
  RK4  : 0.010, 0.013, 0.017, 0.022 s  (explicit CFL limit ~0.026 s at dx=10 m)
  SI   : 2, 5, 10, 15 s               (each ~700–820 s wall; ~50 min total)
  EPI2 : 2, 5, 10, 15 s               (~82 s each regardless of dt)
  EPI3 : 2, 5, 10, 15 s               (~85 s each)

SI is included by default but is slow (~50 min total for all 4 dt values).
Pass --no_si to skip SI and finish in ~30 min total.

Usage
-----
    python experiments/gr_efficiency.py             # full study (~80 min)
    python experiments/gr_efficiency.py --no_si     # skip SI (~30 min)
    python experiments/gr_efficiency.py --plot_only # reload saved results and replot
"""

import argparse
import json
import os
import sys
import time as wall_time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _load_src(name, path):
    """Load a .py source file as a named module without installing the package.

    Executes the source via compile()+exec() so imports are always
    resolved from the live .py file, bypassing any stale .pyc bytecache.
    """
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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid        import Grid
from integrators import step, robert_asselin_filter, shapiro_filter

PLOT_DIR    = "output/figures"
RESULTS_DIR = "output/results"
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

REF_SAVE = os.path.join(RESULTS_DIR, "gr_efficiency_reference.npz")
RES_SAVE = os.path.join(RESULTS_DIR, "gr_efficiency_results.json")


# ---------------------------------------------------------------------------
# G&R Case 2 parameters
# ---------------------------------------------------------------------------

GR = dict(
    Lx=1000., Lz=1000.,   # domain [m]
    dx=10.,   dz=10.,      # grid spacing [m]
    theta_c=0.5,           # bubble amplitude [K]
    rc=250.,               # bubble radius [m]
    xc=500.,               # bubble centre x [m]
    zc=350.,               # bubble centre z [m]
    t_end=700.,            # simulation end time [s]
    dt_ref=0.005,          # reference dt: RK4 at 5 ms  (140 000 steps)
)

# dt values per scheme
DT_STUDY = {
    'RK4':  [0.010, 0.013, 0.017, 0.022],   # stay below CFL limit ~0.026 s
    'SI':   [2.0,   5.0,  10.0,  15.0  ],
    'EPI2': [2.0,   5.0,  10.0,  15.0  ],
    'EPI3': [2.0,   5.0,  10.0,  15.0  ],
}

COLORS = {'RK4': '#1f77b4', 'SI': '#d62728',
          'EPI2': '#2ca02c', 'EPI3': '#ff7f0e'}
MARKERS = {'RK4': 'o', 'SI': 's', 'EPI2': '^', 'EPI3': 'D'}


# ---------------------------------------------------------------------------
# IC helper
# ---------------------------------------------------------------------------

def make_ic(grid):
    """Cosine-bell thermal bubble initial condition (G&R 2008, eq. 3.1)."""
    r = np.sqrt((grid.x_2d - GR['xc'])**2 + (grid.z_2d - GR['zc'])**2)
    state = grid.allocate_state()
    state['theta'] = np.where(
        r <= GR['rc'],
        0.5 * GR['theta_c'] * (1.0 + np.cos(np.pi * r / GR['rc'])),
        0.0
    )
    return state


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------

def compute_or_load_reference():
    """Load reference from disk if it exists, otherwise compute and save it."""
    if os.path.exists(REF_SAVE):
        print(f"  Loading reference from {REF_SAVE}")
        d = np.load(REF_SAVE)
        grid = Grid({'Lx': GR['Lx'], 'Lz': GR['Lz'],
                     'dx': GR['dx'], 'dz': GR['dz']})
        ref = {k: d[k] for k in ['u', 'w', 'theta', 'pi']}
        return ref, grid

    dt_ref  = GR['dt_ref']
    t_end   = GR['t_end']
    n_steps = int(round(t_end / dt_ref))
    print(f"\n  Computing reference: RK4  dt={dt_ref}s  "
          f"steps={n_steps}  T={t_end:.0f}s")
    print(f"  Estimated wall time: ~{n_steps * 3.6e-3 / 60:.0f} min  "
          f"(based on 3.6 ms/step)")

    grid  = Grid({'Lx': GR['Lx'], 'Lz': GR['Lz'],
                  'dx': GR['dx'], 'dz': GR['dz']})
    state = make_ic(grid)
    t0    = wall_time.perf_counter()

    for n in range(n_steps):
        state_new, _, _ = step(state, grid, dt_ref, scheme='RK4')
        state = state_new
        if not np.all(np.isfinite(state['u'])):
            print("  Reference BLOW-UP"); return None, grid

    elapsed = wall_time.perf_counter() - t0
    print(f"  Reference done in {elapsed:.1f}s")

    np.savez(REF_SAVE, **state)
    print(f"  Saved to {REF_SAVE}")
    return state, grid


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def l2_error(ref, test, variables=('theta', 'w')):
    """
    Combined relative L2 error across listed variables.
    L2 = sqrt(sum_v ||ref_v - test_v||^2) / sqrt(sum_v ||ref_v||^2 + eps)
    """
    num = sum(np.sum((ref[v] - test[v])**2) for v in variables)
    den = sum(np.sum(ref[v]**2)             for v in variables) + 1e-30
    return float(np.sqrt(num / den))


# ---------------------------------------------------------------------------
# Run one scheme at one dt
# ---------------------------------------------------------------------------

def run_one(scheme, dt, grid, verbose=True):
    """
    Integrate G&R Case 2 from t=0 to t_end with (scheme, dt).
    Returns (state_final, wall_time_total).  Returns (None, wall) on blow-up.
    """
    t_end   = GR['t_end']
    n_steps = int(round(t_end / dt))
    is_epi  = scheme in ('EPI2', 'EPI3')

    state      = make_ic(grid)
    state_old  = None
    epi_n_prev = None

    t0    = wall_time.perf_counter()
    blown = False

    for n in range(n_steps):
        state_old_saved = state_old
        try:
            state_new, state_old, epi_extra = step(
                state, grid, dt, scheme=scheme,
                state_old=state_old, epi_n_prev=epi_n_prev
            )
        except Exception as e:
            if verbose:
                print(f"    Exception at step {n+1}: {e}")
            blown = True; break

        if scheme == 'CTCS' and n > 0:
            state = robert_asselin_filter(state_old_saved, state,
                                          state_new, alpha=0.1)
        else:
            state = state_new

        if is_epi and epi_extra is not None:
            epi_n_prev = epi_extra['n_rhs']

        # No Shapiro filter for G&R Case 2 (stable without it)

        if not np.all(np.isfinite(state['u'])):
            blown = True; break

    elapsed = wall_time.perf_counter() - t0
    if blown:
        return None, elapsed
    return state, elapsed


# ---------------------------------------------------------------------------
# Full efficiency study
# ---------------------------------------------------------------------------

def run_study(schemes=('RK4', 'SI', 'EPI2', 'EPI3')):
    """Run all (scheme, dt) combinations and return results dict."""
    ref, grid = compute_or_load_reference()
    if ref is None:
        return {}

    results = {}   # results[scheme][dt_str] = {error, wall_s, n_steps}

    for scheme in schemes:
        results[scheme] = {}
        dts = DT_STUDY[scheme]
        n_total = len(dts)
        print(f"\n  Scheme: {scheme}  ({n_total} dt values)")

        for i, dt in enumerate(dts):
            n_steps = int(round(GR['t_end'] / dt))
            print(f"    [{i+1}/{n_total}]  dt={dt:.3f}s  steps={n_steps} ...",
                  end=' ', flush=True)

            state_f, elapsed = run_one(scheme, dt, grid, verbose=False)

            if state_f is None:
                print(f"BLOW-UP  ({elapsed:.1f}s)")
                continue

            err = l2_error(ref, state_f)
            print(f"L2={err:.3e}  wall={elapsed:.1f}s")

            results[scheme][str(dt)] = {
                'dt':       dt,
                'error_l2': err,
                'wall_s':   elapsed,
                'n_steps':  n_steps,
            }

    # Save results to JSON for later replotting
    with open(RES_SAVE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RES_SAVE}")
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_efficiency(results, filename=None):
    """
    Error vs wall-clock time — the key dissertation figure.

    - x-axis: total wall-clock time (log scale, seconds)
    - y-axis: relative L2 error in (θ', w) (log scale)
    - Each scheme is a line; each point is one dt value
    - Slope guides show 1st, 2nd, 4th order cost-accuracy trade-off
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for scheme, data in results.items():
        if not data:
            continue
        pts = sorted(data.values(), key=lambda d: d['wall_s'])
        xs = [p['wall_s']   for p in pts]
        ys = [p['error_l2'] for p in pts]
        dts = [p['dt']       for p in pts]

        ax.plot(xs, ys, '-', color=COLORS[scheme],
                marker=MARKERS[scheme], markersize=7, linewidth=1.5,
                label=scheme, zorder=3)

        # Annotate each point with its dt value
        for x, y, dt in zip(xs, ys, dts):
            ax.annotate(f'Δt={dt:.3g}s', (x, y),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=7, color=COLORS[scheme])

    # Slope guide lines (reference triangles)
    # Place guides near top-left of data
    all_xs = [p['wall_s']   for d in results.values() for p in d.values()]
    all_ys = [p['error_l2'] for d in results.values() for p in d.values()]
    if all_xs:
        xg = min(all_xs) * 1.5
        yg = max(all_ys) * 0.3
        for order, label, dy in [(1, '1st', 1), (2, '2nd', 2), (4, '4th', 4)]:
            # slope line: as x doubles, y decreases by 2^order
            xs_g = np.array([xg, xg * 4])
            ys_g = yg * np.array([1, 4**(-order)])
            ax.plot(xs_g, ys_g, 'k--', linewidth=0.7, alpha=0.4)
            ax.text(xs_g[-1] * 1.05, ys_g[-1], f'∝t$^{{-{order}}}$',
                    fontsize=8, alpha=0.6, va='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wall-clock time (s)', fontsize=11)
    ax.set_ylabel('Relative L₂ error  (θ\', w)', fontsize=11)
    ax.set_title('Efficiency: G&R (2008) Case 2  —  t = 700 s,  dx = 10 m',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if filename is None:
        filename = os.path.join(PLOT_DIR, 'gr_efficiency.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {filename}")
    return filename


def plot_convergence(results, filename=None):
    """
    Error vs dt — shows convergence order of each scheme.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for scheme, data in results.items():
        if not data:
            continue
        pts = sorted(data.values(), key=lambda d: d['dt'])
        xs = [p['dt']       for p in pts]
        ys = [p['error_l2'] for p in pts]

        ax.plot(xs, ys, '-', color=COLORS[scheme],
                marker=MARKERS[scheme], markersize=7, linewidth=1.5,
                label=scheme, zorder=3)

    # Reference slopes
    if any(results.values()):
        all_xs = [p['dt']       for d in results.values() for p in d.values()]
        all_ys = [p['error_l2'] for d in results.values() for p in d.values()]
        xr = max(all_xs) * 0.6
        yr = max(all_ys) * 0.5
        for order, label in [(1, '1st'), (2, '2nd'), (4, '4th')]:
            xs_r = np.array([xr / 4, xr])
            ys_r = yr * np.array([4**(-order), 1.0])
            ax.plot(xs_r, ys_r, 'k--', linewidth=0.7, alpha=0.4)
            ax.text(xs_r[0] * 0.9, ys_r[0], f'{label}',
                    fontsize=8, alpha=0.6, ha='right', va='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Δt (s)', fontsize=11)
    ax.set_ylabel('Relative L₂ error  (θ\', w)', fontsize=11)
    ax.set_title('Convergence: G&R (2008) Case 2  —  t = 700 s,  dx = 10 m',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if filename is None:
        filename = os.path.join(PLOT_DIR, 'gr_convergence.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {filename}")
    return filename


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G&R Case 2 efficiency study"
    )
    parser.add_argument('--no_si',    action='store_true',
                        help='Skip SI (saves ~50 min)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Load saved results and replot (no runs)')
    parser.add_argument('--schemes',  nargs='+',
                        default=['RK4', 'SI', 'EPI2', 'EPI3'],
                        help='Schemes to run (default: all four)')
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(RES_SAVE):
            print(f"No saved results at {RES_SAVE}. Run without --plot_only first.")
            return
        with open(RES_SAVE) as f:
            results = json.load(f)
        # JSON loads dt keys as strings; convert to numeric for sorting
        for scheme in results:
            results[scheme] = {
                k: v for k, v in results[scheme].items()
            }
        plot_efficiency(results)
        plot_convergence(results)
        return

    schemes = args.schemes
    if args.no_si and 'SI' in schemes:
        schemes = [s for s in schemes if s != 'SI']
        print("  Skipping SI (--no_si)")

    print(f"\n{'='*60}")
    print("  G&R (2008) Case 2 — Efficiency Study")
    print(f"  Domain: {GR['Lx']:.0f}×{GR['Lz']:.0f} m  "
          f"dx={GR['dx']:.0f} m  t_end={GR['t_end']:.0f} s")
    print(f"  Reference: RK4 dt={GR['dt_ref']} s")
    print(f"  Schemes: {schemes}")
    if 'SI' in schemes:
        print("  NOTE: SI will take ~50 min. Use --no_si to skip.")
    print(f"{'='*60}")

    results = run_study(schemes=schemes)
    if results:
        plot_efficiency(results)
        plot_convergence(results)

        # Print summary table
        print(f"\n{'='*60}")
        print(f"  {'Scheme':6s}  {'dt':8s}  {'steps':7s}  "
              f"{'wall(s)':9s}  {'L2 error':12s}")
        print('-' * 60)
        for scheme in schemes:
            for dt_str, r in sorted(results.get(scheme, {}).items(),
                                     key=lambda x: float(x[0])):
                print(f"  {scheme:6s}  {r['dt']:8.3f}  {r['n_steps']:7d}  "
                      f"{r['wall_s']:9.1f}  {r['error_l2']:12.3e}")
        print('=' * 60)


if __name__ == "__main__":
    main()
