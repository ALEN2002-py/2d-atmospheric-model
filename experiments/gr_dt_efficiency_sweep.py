"""
gr_dt_efficiency_sweep.py
==========================
G&R Case 2: the real (measured, not hardcoded) efficiency comparison across
RK4, SI, and SI2 -- dt vs wall-time vs accuracy, all at dx=10 m.

This closes out the "dt(SI/SI2) vs dt(RK4)" and "wall time SI vs RK4" items
from the supervisor meeting list. Unlike plot_efficiency_frontier.py (whose
SI/EPI points are hardcoded from CLAUDE.md), every point here is either
freshly measured (RK4) or pulled from a real completed run
(output/results/gmres_perf_dx10m.json, produced by gr_gmres_performance.py).

Blown-up runs (SI dt=4,8) are excluded from the accuracy comparison --
their theta_max reflects a diverging solution, not truncation error.

USAGE
-----
  python experiments/gr_dt_efficiency_sweep.py
  python experiments/gr_dt_efficiency_sweep.py --skip-rk4   # reuse cached RK4 only
"""

import argparse
import json
import os
import sys
import time as wall_time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def _load_src(name, path):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    exec(compile(source, os.path.abspath(path), "exec"), mod.__dict__)
    return mod

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_load_src("grid",        os.path.join(_src, "grid.py"))
_load_src("dynamics",    os.path.join(_src, "dynamics.py"))
_load_src("integrators", os.path.join(_src, "integrators.py"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid        import Grid
from integrators import step

OUT_DIR      = "output/figures"
RESULTS_DIR  = "output/results"
GMRES_CACHE  = os.path.join(RESULTS_DIR, "gmres_perf_dx10m.json")
RK4_CACHE    = os.path.join(RESULTS_DIR, "rk4_dt_sweep_dx10m.json")
EPI2V_CACHE  = os.path.join(RESULTS_DIR, "epi2v_dt_sweep_dx10m.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Internal scheme keys ("EPI2V") are unchanged; this only renames the
# thesis-facing plot labels to match the dissertation's ETD1/ETD1V terminology.
DISPLAY_NAME = {"EPI2V": "ETD1V"}

# EPI2V has no acoustic-CFL restriction (unconditionally stable linear part),
# so this range deliberately spans from RK4-comparable short steps up past
# where SI/SI2 already struggle -- per Dr. Clancy's request to see the full
# shape of the curve, not just the long-dt regime. Because EPI2V's sub-step
# count p scales linearly with dt (p ~ dt), total Krylov work per run is
# roughly dt-INDEPENDENT (n_steps ~ 1/dt, p ~ dt, product ~ const) -- so,
# unlike RK4, sweeping small dt here is not disproportionately expensive.
EPI2V_DT_VALS = [0.02, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]

# ---------------------------------------------------------------------------
# G&R Case 2 parameters
# ---------------------------------------------------------------------------
LX, LZ  = 1000.0, 1000.0
THETA_C = 0.5
R_C     = 250.0
X_C, Z_C = 500.0, 350.0
T_END    = 700.0
DX       = 10.0
THETA_REF = 0.570   # K -- G&R (2008) Table 3, SE models, 5 m resolution

RK4_DT_VALS = [0.02, 0.01]   # dt=0.02 CFL~0.69 (near limit), dt=0.01 CFL~0.35 (reference)


_kappa_fallback_warned = False   # module-level: print the warning only once per run


def _kappa_used(dt, dx=10.0, kappa_requested=200.0):
    """
    Recompute the auto-capped order-4 diffusion coefficient actually used by
    gr_gmres_performance.py for a given dt (same formula as that script's
    run_scheme()). This is ONLY a fallback for cache entries predating the
    'kappa_used' field -- kappa_requested here is an ASSUMPTION (matching
    gr_gmres_performance.py's --kappa default of 200.0), not something read
    from the cache, so it silently gives the wrong answer if that script was
    ever run with a different --kappa. Prints a one-time visible warning
    whenever it's actually invoked, precisely so that assumption can't fail
    silently.

    Also used to annotate how much the SI/SI2 accuracy trend vs dt is
    confounded by the diffusion coefficient shrinking at larger dt (less
    damping at large dt pulls theta_max toward the reference for reasons
    unrelated to genuine temporal truncation error).
    """
    global _kappa_fallback_warned
    if not _kappa_fallback_warned:
        print(f"  WARNING: one or more cache entries are missing 'kappa_used' -- "
              f"assuming --kappa={kappa_requested} (gr_gmres_performance.py's "
              f"default) for all of them. Re-run that script to refresh the "
              f"cache with the real value(s) if it was ever run with a "
              f"different --kappa.")
        _kappa_fallback_warned = True
    eig_max_o4 = (8.0 / dx**2) ** 2
    kappa_stable = 0.7 * 2.0 / (eig_max_o4 * dt) if dt > 0 else kappa_requested
    return min(kappa_requested, kappa_stable)


def make_ic(grid):
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X_C)**2 + (grid.z_2d - Z_C)**2)
    state["theta"] = np.where(
        r <= R_C, 0.5 * THETA_C * (1.0 + np.cos(np.pi * r / R_C)), 0.0
    )
    return state


def run_rk4(dt):
    grid  = Grid({"Lx": LX, "Lz": LZ, "dx": DX, "dz": DX})
    state = make_ic(grid)
    n_steps = int(round(T_END / dt))
    t0 = wall_time.perf_counter()
    for _ in range(n_steps):
        state, _, _ = step(state, grid, dt, scheme="RK4")
    wall = wall_time.perf_counter() - t0
    return {
        "wall_time": wall,
        "theta_max": float(np.max(state["theta"])),
        "w_max":     float(np.max(np.abs(state["w"]))),
        "blew_up":   bool(not np.isfinite(state["theta"]).all()),
    }


def _run_rk4_worker(dt):
    """Module-level (picklable) wrapper so each RK4 dt runs in its own
    process via ProcessPoolExecutor -- the dt points are fully independent."""
    return dt, run_rk4(dt)


def get_rk4_results(skip_run):
    if os.path.exists(RK4_CACHE):
        with open(RK4_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  Loaded cached RK4 results from {RK4_CACHE}")
        return cached
    if skip_run:
        print("  --skip-rk4 and no cache found -- RK4 points will be omitted")
        return {}

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    print(f"  Running {len(RK4_DT_VALS)} RK4 dt value(s) in parallel...", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=len(RK4_DT_VALS)) as executor:
        futures = {executor.submit(_run_rk4_worker, dt): dt for dt in RK4_DT_VALS}
        for fut in as_completed(futures):
            dt, r = fut.result()
            results[str(dt)] = r
            print(f"    dt={dt}s: theta_max={r['theta_max']:.3f}K  "
                  f"w_max={r['w_max']:.3f}m/s  wall={r['wall_time']:.1f}s", flush=True)

    with open(RK4_CACHE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached RK4 results -> {RK4_CACHE}")
    return results


def run_epi2v(dt):
    grid  = Grid({"Lx": LX, "Lz": LZ, "dx": DX, "dz": DX})
    state = make_ic(grid)
    n_steps = int(round(T_END / dt))
    t0 = wall_time.perf_counter()
    for _ in range(n_steps):
        state, _, _ = step(state, grid, dt, scheme="EPI2V")
        if not np.isfinite(state["theta"]).all() or np.abs(state["theta"]).max() > 50.0:
            break
    wall = wall_time.perf_counter() - t0
    return {
        "wall_time": wall,
        "theta_max": float(np.nanmax(state["theta"])),
        "w_max":     float(np.nanmax(np.abs(state["w"]))),
        "blew_up":   bool(not np.isfinite(state["theta"]).all()),
    }


def _run_epi2v_worker(dt):
    """Module-level (picklable) wrapper so each EPI2V dt runs in its own
    process via ProcessPoolExecutor -- the dt points are fully independent."""
    return dt, run_epi2v(dt)


def get_epi2v_results(skip_run):
    if os.path.exists(EPI2V_CACHE):
        with open(EPI2V_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  Loaded cached EPI2V results from {EPI2V_CACHE}")
        return cached
    if skip_run:
        print("  --skip-epi2v and no cache found -- EPI2V points will be omitted")
        return {}

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    n_workers = min(len(EPI2V_DT_VALS), os.cpu_count() or 1)
    print(f"  Running {len(EPI2V_DT_VALS)} EPI2V dt value(s) across "
          f"{n_workers} worker process(es) in parallel...", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_epi2v_worker, dt): dt for dt in EPI2V_DT_VALS}
        for fut in as_completed(futures):
            dt, r = fut.result()
            results[str(dt)] = r
            blow = "  *** BLOW-UP ***" if r["blew_up"] else ""
            print(f"    dt={dt}s: theta_max={r['theta_max']:.3f}K  "
                  f"w_max={r['w_max']:.3f}m/s  wall={r['wall_time']:.1f}s{blow}", flush=True)

    with open(EPI2V_CACHE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached EPI2V results -> {EPI2V_CACHE}")
    return results


def load_gmres_cache():
    if not os.path.exists(GMRES_CACHE):
        raise FileNotFoundError(
            f"{GMRES_CACHE} not found -- run "
            f"'python experiments/gr_gmres_performance.py --scheme both' first."
        )
    with open(GMRES_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_series(rk4_results, gmres_results, scheme, epi2v_results=None):
    """Return (dt_vals, wall_times, errors) sorted by dt, blown-up points excluded."""
    pts = []
    if scheme == "RK4":
        src = rk4_results
    elif scheme == "EPI2V":
        src = epi2v_results or {}
    else:
        src = gmres_results.get(scheme, {})
    for dt_str, r in src.items():
        if r.get("blew_up"):
            continue
        err = abs(r["theta_max"] - THETA_REF) / THETA_REF * 100.0
        pts.append((float(dt_str), r["wall_time"], err))
    pts.sort(key=lambda p: p[0])
    if not pts:
        return [], [], []
    dts, walls, errs = zip(*pts)
    return list(dts), list(walls), list(errs)


def make_frontier_plot(rk4_results, gmres_results, epi2v_results=None):
    """Line plot (one line per scheme, points ordered by dt) instead of a
    scattered/annotated cloud -- much easier to read the dt progression and
    compare schemes than dt-labelled text scattered next to each marker."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    styles = {
        "RK4":   {"color": "#d62728", "marker": "o"},
        "SI":    {"color": "#9467bd", "marker": "s"},
        "SI2":   {"color": "#2ca02c", "marker": "^"},
        "SI2LU": {"color": "#1f77b4", "marker": "D"},
        "EPI2V": {"color": "#ff7f0e", "marker": "P"},
    }

    for i, scheme in enumerate(("RK4", "SI2", "EPI2V")):
        dts, walls, errs = _collect_series(rk4_results, gmres_results, scheme, epi2v_results)
        if not walls:
            continue
        s = styles[scheme]
        ax.plot(walls, errs, "-", color=s["color"], lw=1.6, alpha=0.75, zorder=3)
        ax.scatter(walls, errs, s=95, color=s["color"], marker=s["marker"],
                   edgecolor="white", linewidth=0.8, zorder=5,
                   label=f"{DISPLAY_NAME.get(scheme, scheme)} (measured)")

        # Label every point with just its dt -- alternate above/below by
        # index so labels from a tightly-packed scheme don't stack directly
        # on top of one another; no kappa clutter (that's in the table).
        for j, (dt, w, e) in enumerate(zip(dts, walls, errs)):
            dy = 9 if j % 2 == 0 else -13
            va = "bottom" if j % 2 == 0 else "top"
            ax.annotate(f"{dt:g}s", xy=(w, e), xytext=(0, dy),
                        textcoords="offset points", ha="center", va=va,
                        fontsize=7.5, color=s["color"])

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time [s]  (log scale)", fontsize=12)
    ax.set_ylabel(r"Error in $\theta'_{\max}$ relative to G&R ref (5 m) [%]", fontsize=11)
    ax.set_title(
        "Measured Efficiency Frontier — G&R Case 2 (dx=10 m, t_end=700 s)\n"
        "Lines connect points of the same scheme in dt order (labelled). "
        "Blown-up runs excluded.",
        fontsize=11,
    )
    ax.axhline(y=0, color="#aaa", lw=0.7, ls="--")
    ax.grid(True, which="both", color="#e8e8e8", lw=0.5)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#ccc")
    ax.margins(y=0.15)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gr_dt_efficiency_frontier_measured.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


def make_relative_time_plot(rk4_results, gmres_results, epi2v_results):
    """
    The plot Dr. Clancy sketched on the whiteboard: wall time for each
    scheme, at each dt, divided by RK4's OWN best (fastest stable) wall
    time -- so RK4's best point sits at y=1 by construction, and every
    other point shows how many times slower (>1) or faster (<1) than
    RK4's own optimum that configuration is.

    His point in sketching this: SI/SI2 trace a bathtub/U-shape (cost high
    at both very small and very large dt, cheapest somewhere in the middle),
    while EPI, having no acoustic-CFL restriction, can be pushed to much
    larger dt than SI/SI2 before its own curve turns back upward.
    """
    rk4_stable = [r["wall_time"] for r in rk4_results.values() if not r.get("blew_up")]
    if not rk4_stable:
        print("  Skipping relative-time plot -- no stable RK4 baseline point available.")
        return
    baseline = min(rk4_stable)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    styles = {
        "RK4":   {"color": "#d62728", "marker": "o"},
        "SI":    {"color": "#9467bd", "marker": "s"},
        "SI2":   {"color": "#2ca02c", "marker": "^"},
        "SI2LU": {"color": "#1f77b4", "marker": "D"},
        "EPI2V": {"color": "#ff7f0e", "marker": "P"},
    }

    any_plotted = False
    for scheme in ("RK4", "SI2", "EPI2V"):
        dts, walls, _ = _collect_series(rk4_results, gmres_results, scheme, epi2v_results)
        if not walls:
            continue
        any_plotted = True
        rel = [w / baseline for w in walls]
        s = styles[scheme]
        ax.plot(dts, rel, "-", color=s["color"], lw=1.8, alpha=0.8, zorder=3)
        ax.scatter(dts, rel, s=95, color=s["color"], marker=s["marker"],
                   edgecolor="white", linewidth=0.8, zorder=5,
                   label=f"{DISPLAY_NAME.get(scheme, scheme)} (measured)")

    ax.axhline(y=1.0, color="#888", lw=1.0, ls="--", zorder=1,
               label="RK4's own best dt (baseline = 1)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta t$ [s]  (log scale)", fontsize=12)
    ax.set_ylabel("Wall time, relative to RK4's own best dt  (log scale)", fontsize=11)
    ax.set_title(
        "Relative Cost vs. $\\Delta t$ — G&R Case 2 (dx=10 m, t_end=700 s)\n"
        "Reproduces the whiteboard sketch: SI/SI2 trace a bathtub curve, "
        "ETD1V extends further right before turning up.",
        fontsize=11,
    )
    ax.grid(True, which="both", color="#e8e8e8", lw=0.5)
    if any_plotted:
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gr_relative_time_vs_dt.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


def make_error_vs_dt_plot(rk4_results, gmres_results, epi2v_results):
    """
    The OTHER half of Dr. Clancy's whiteboard sketch (its top panel): error
    plotted directly against dt, rather than against wall time as the
    efficiency frontier (make_frontier_plot) does. This is a plain
    convergence-style plot -- one line per scheme, error % vs dt on a log-x
    axis -- and is what actually shows whether a scheme's accuracy is
    genuinely dt-converged (flattening at small dt) as opposed to just
    looking good at a particular dt for confounded reasons (see the
    kappa-vs-dt note printed by print_summary_table).
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    styles = {
        "RK4":   {"color": "#d62728", "marker": "o"},
        "SI":    {"color": "#9467bd", "marker": "s"},
        "SI2":   {"color": "#2ca02c", "marker": "^"},
        "SI2LU": {"color": "#1f77b4", "marker": "D"},
        "EPI2V": {"color": "#ff7f0e", "marker": "P"},
    }

    any_plotted = False
    for scheme in ("RK4", "SI2", "EPI2V"):
        dts, _, errs = _collect_series(rk4_results, gmres_results, scheme, epi2v_results)
        if not errs:
            continue
        any_plotted = True
        s = styles[scheme]
        ax.plot(dts, errs, "-", color=s["color"], lw=1.8, alpha=0.8, zorder=3)
        ax.scatter(dts, errs, s=95, color=s["color"], marker=s["marker"],
                   edgecolor="white", linewidth=0.8, zorder=5,
                   label=f"{DISPLAY_NAME.get(scheme, scheme)} (measured)")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta t$ [s]  (log scale)", fontsize=12)
    ax.set_ylabel(r"Error in $\theta'_{\max}$ relative to G&R ref (5 m) [%]", fontsize=11)
    ax.set_title(
        "Error vs. $\\Delta t$ — G&R Case 2 (dx=10 m, t_end=700 s)\n"
        "Top panel of the whiteboard sketch: does error genuinely converge "
        "as $\\Delta t$ shrinks, or just move for confounded reasons?",
        fontsize=11,
    )
    ax.axhline(y=0, color="#aaa", lw=0.7, ls="--")
    ax.grid(True, which="both", color="#e8e8e8", lw=0.5)
    if any_plotted:
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gr_error_vs_dt.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_summary_table(rk4_results, gmres_results, epi2v_results=None):
    print("\n" + "=" * 100)
    print("  MEASURED EFFICIENCY COMPARISON -- G&R Case 2, dx=10 m, t_end=700 s")
    print("=" * 100)
    print(f"  {'Scheme':<6} {'dt(s)':>8} {'wall(s)':>10} {'theta_max(K)':>13} "
          f"{'err_vs_ref(%)':>14} {'kappa4_used':>12} {'blowup':>8}")
    print("  " + "-" * 96)

    def _row(scheme, dt_str, r, kappa=None):
        blow = "YES" if r.get("blew_up") else "no"
        if r.get("blew_up"):
            # Diverging solution -- theta_max reflects the blow-up, not
            # truncation error, so the error % is meaningless here.
            err_str = "N/A (diverged)"
        else:
            err = abs(r["theta_max"] - THETA_REF) / THETA_REF * 100.0
            err_str = f"{err:.1f}"
        kappa_str = f"{kappa:.0f}" if kappa is not None else "0 (none)"
        print(f"  {scheme:<6} {dt_str:>8} {r['wall_time']:>10.1f} "
              f"{r['theta_max']:>13.3f} {err_str:>14} {kappa_str:>12} {blow:>8}")

    for dt_str, r in rk4_results.items():
        _row("RK4", dt_str, r, kappa=None)
    for scheme in ("SI", "SI2", "SI2LU"):
        for dt_str, r in gmres_results.get(scheme, {}).items():
            kappa = r.get("kappa_used")
            if kappa is None:
                kappa = _kappa_used(float(dt_str))
            _row(scheme, dt_str, r, kappa=kappa)
    for dt_str, r in (epi2v_results or {}).items():
        _row("EPI2V", dt_str, r, kappa=None)
    print("=" * 100)
    print(f"  Reference: G&R (2008) Table 3, SE models, 5 m resolution: "
          f"theta'_max = {THETA_REF} K")
    print("  NOTE: kappa4_used shrinks automatically at larger dt (stability cap) --")
    print("  SI/SI2 accuracy improving with dt partly reflects less damping, not")
    print("  purely improving temporal truncation error. See discussion in text.\n")
    if not any(gmres_results.get(s) for s in ("SI", "SI2")):
        print("  NOTE: the SI/SI2 cache currently only has SI2LU entries -- SI/SI2 rows")
        print("  and plot series are empty until gr_gmres_performance.py --scheme both")
        print("  is (re-)run. Not an EPI2V-related gap.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real (measured) dt/wall-time/accuracy comparison: RK4 vs SI vs SI2 vs EPI2V"
    )
    parser.add_argument("--skip-rk4", action="store_true",
                        help="Don't run RK4 fresh; use cache only (omit if no cache)")
    parser.add_argument("--skip-epi2v", action="store_true",
                        help="Don't run EPI2V fresh; use cache only (omit if no cache)")
    args = parser.parse_args()

    gmres_results = load_gmres_cache()
    rk4_results   = get_rk4_results(args.skip_rk4)
    epi2v_results = get_epi2v_results(args.skip_epi2v)

    print_summary_table(rk4_results, gmres_results, epi2v_results)
    make_frontier_plot(rk4_results, gmres_results, epi2v_results)
    make_relative_time_plot(rk4_results, gmres_results, epi2v_results)
    make_error_vs_dt_plot(rk4_results, gmres_results, epi2v_results)


if __name__ == "__main__":
    main()
