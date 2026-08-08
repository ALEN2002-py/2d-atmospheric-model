"""
pc_gmres_performance.py
========================
GMRES performance analysis for SI and SI2 schemes on P&C (2022) Experiment 1.

P&C analogue of gr_gmres_performance.py. Tracks GMRES iteration count at
every time step for multiple dt values on the 5km P&C domain (250x250 grid
at the paper's dx=20m -- 6.25x more unknowns per GMRES solve than G&R
Case 2's 100x100 grid, so expect this to run considerably longer).

Every (scheme, dt) pair is a fully independent simulation, so they run
concurrently via ProcessPoolExecutor (--workers to override the pool size,
default: one worker per job capped at CPU count) -- same pattern as
pc_si_diffusion_comparison.py.

Produces three plots per scheme:
  1. Iteration count vs simulation time  (all dt on one figure)
  2. Mean & max GMRES iterations vs dt
  3. Wall time breakdown (GMRES cost vs dt)

Usage
-----
    python experiments/pc_gmres_performance.py              # SI only
    python experiments/pc_gmres_performance.py --scheme SI2
    python experiments/pc_gmres_performance.py --scheme both
    python experiments/pc_gmres_performance.py --scheme both --plot-from-cache

Output: output/figures/pc_gmres_*.png
"""

import argparse
import math
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

_src = os.path.join(os.path.dirname(__file__), "..", "src")
_load_src("grid",        os.path.join(_src, "grid.py"))
_load_src("dynamics",    os.path.join(_src, "dynamics.py"))
_load_src("integrators", os.path.join(_src, "integrators.py"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid        import Grid
from integrators import step, robert_asselin_filter_time

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# P&C Exp 1 parameters (Section 6.1)
# ---------------------------------------------------------------------------
PAPER = {
    "Lx":    5000.0,
    "Lz":    5000.0,
    "AT":       0.5,
    "a":      400.0,
    "sigma":  100.0,
    "x0":    2500.0,
    "z0":     700.0,
    # 900s (not the full 1800s paper duration) -- enough to pass the
    # buoyancy timescale (3T~660s, where velocity peaks) while keeping the
    # GMRES sweep affordable; this script characterises solver cost scaling
    # with dt, not whether the full paper run is sustainable (that's
    # pc_si_diffusion_comparison.py's job).
    "t_end":  900.0,
}

def make_state(grid):
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - PAPER["x0"])**2 + (grid.z_2d - PAPER["z0"])**2)
    a, sigma, AT = PAPER["a"], PAPER["sigma"], PAPER["AT"]
    state["theta"] = np.where(
        r <= a, AT, AT * np.exp(-(r - a)**2 / (2.0 * sigma**2)),
    )
    return state


def run_scheme(scheme, dx, dt, t_end, kappa=0.0, verbose=True):
    """
    Run SI or SI2 for the P&C Exp 1 bubble.
    Returns dict with step_times, gmres_iters, wall_time, blew_up,
    theta_max, w_max, kappa_used.
    """
    # Scale kappa (order-4) to stay within the explicit forward-Euler
    # diffusion stability limit for THIS dt: kappa * eig_max * dt <= 2 --
    # see gr_gmres_performance.py for the full derivation/rationale.
    eig_max_o4   = (8.0 / dx**2) ** 2
    kappa_stable = 0.7 * 2.0 / (eig_max_o4 * dt) if dt > 0 else kappa
    kappa_used   = min(kappa, kappa_stable) if kappa > 0 else 0.0

    g = Grid({"Lx": PAPER["Lx"], "Lz": PAPER["Lz"], "dx": dx, "dz": dx,
              "stratification": "isentropic",
              "diffusion_coeff": kappa_used,
              "diffusion_order": 4})

    state     = make_state(g)
    state_old = None
    n_steps   = int(round(t_end / dt))

    step_times  = []
    gmres_iters = []

    t0 = wall_time.time()
    blew_up = False

    for n in range(n_steps):
        t_now = n * dt
        try:
            state_new, state_prev, extra = step(
                state, g, dt, scheme=scheme, state_old=state_old,
            )
        except Exception as e:
            if verbose:
                print(f"  {scheme} dt={dt:.2f}s: blew up at t={t_now:.1f}s -- {e}")
            blew_up = True
            break

        # Blow-up: NaN/Inf, or finite-but-unphysical magnitude (forward-Euler
        # divergence can grow to huge finite float64 values for many steps
        # before ever producing a NaN -- isfinite() alone would miss it).
        # Threshold set well above the bubble amplitude (AT=0.5K).
        if not np.isfinite(state_new["theta"]).all() or \
           np.abs(state_new["theta"]).max() > 50.0:
            if verbose:
                print(f"  {scheme} dt={dt:.2f}s: blew up at t={t_now:.1f}s "
                      f"(theta_max={np.nanmax(np.abs(state_new['theta'])):.3g}K)")
            blew_up = True
            break

        iters = extra["gmres_iters"] if extra and "gmres_iters" in extra else 0
        step_times.append(t_now + dt)
        gmres_iters.append(iters)

        if scheme in ("SI2", "SI2LU") and state_old is not None:
            state_filtered = robert_asselin_filter_time(state_old, state, state_new)
        else:
            state_filtered = state

        state_old = state_filtered
        state     = state_new

    wall = wall_time.time() - t0

    theta_max = float(np.max(state["theta"])) if len(step_times) else 0.0
    w_max     = float(np.max(np.abs(state["w"]))) if len(step_times) else 0.0

    if verbose and not blew_up:
        iters_arr = np.array(gmres_iters)
        print(f"  {scheme} dt={dt:.2f}s: {len(step_times)} steps, "
              f"GMRES mean={iters_arr.mean():.1f} max={iters_arr.max():.0f}, "
              f"wall={wall:.1f}s, theta_max={theta_max:.3f}K, w_max={w_max:.3f}m/s")

    return {
        "step_times":  np.array(step_times),
        "gmres_iters": np.array(gmres_iters, dtype=float),
        "wall_time":   wall,
        "blew_up":     blew_up,
        "theta_max":   theta_max,
        "w_max":       w_max,
        "kappa_used":  kappa_used,
    }


def plot_iters_vs_time(results_by_dt, scheme, dx):
    """
    SI2LU only ever calls GMRES once per run (the step-0 bootstrap, before
    switching to the direct LU solve) -- every later step is exactly 0, not
    just small. A line chart of that is a single-pixel spike at t=dt on a
    900s axis (often hidden under the legend) followed by flat zero --
    technically correct but unreadable. Use a bar chart of the one real
    number (bootstrap iterations per dt) instead. See the equivalent fix in
    gr_gmres_performance.py.
    """
    dt_vals = sorted(results_by_dt.keys())

    if scheme.upper() == "SI2LU":
        boot_iters, labels = [], []
        for dt in dt_vals:
            r = results_by_dt[dt]
            if len(r["gmres_iters"]) == 0:
                continue
            boot_iters.append(r["gmres_iters"][0])
            labels.append(f"dt={dt:.2f}s")

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, boot_iters, color="#1f77b4")
        for b, v in zip(bars, boot_iters):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("GMRES iterations (one-time bootstrap step only)", fontsize=12)
        ax.set_title(f"{scheme}: one-time bootstrap GMRES cost -- P&C Exp 1, dx={dx:.0f} m\n"
                      "Every step after the first uses the direct LU solve: 0 GMRES iterations",
                      fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"pc_gmres_iters_vs_time_{scheme.lower()}_dx{dx:.0f}m.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    cmap    = plt.cm.viridis
    colors  = [cmap(i / max(len(dt_vals) - 1, 1)) for i in range(len(dt_vals))]

    for dt, c in zip(dt_vals, colors):
        r = results_by_dt[dt]
        if r["blew_up"] and len(r["step_times"]) == 0:
            continue
        lbl = f"dt={dt:.2f}s" + (" [blew up]" if r["blew_up"] else "")
        ax.plot(r["step_times"], r["gmres_iters"],
                color=c, lw=1.2, alpha=0.8, label=lbl)

    ax.set_xlabel("Simulation time [s]", fontsize=12)
    ax.set_ylabel("GMRES iterations per step", fontsize=12)
    ax.set_title(f"{scheme}: GMRES iterations vs time -- P&C Exp 1, dx={dx:.0f} m",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, PAPER["t_end"])

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"pc_gmres_iters_vs_time_{scheme.lower()}_dx{dx:.0f}m.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_mean_max_vs_dt(results_by_dt, scheme, dx):
    """Blown-up dt values are excluded entirely: a run that diverged partway
    only has GMRES statistics from the steps before it blew up, not
    comparable to a completed run -- see gr_gmres_performance.py."""
    dt_vals   = sorted(results_by_dt.keys())
    valid_dts, means, maxs = [], [], []

    for dt in dt_vals:
        r = results_by_dt[dt]
        if len(r["gmres_iters"]) == 0 or r["blew_up"]:
            continue
        valid_dts.append(dt)
        means.append(r["gmres_iters"].mean())
        maxs.append(r["gmres_iters"].max())

    _draw_mean_max_vs_dt(valid_dts, means, maxs, scheme, dx)


def _draw_mean_max_vs_dt(valid_dts, means, maxs, scheme, dx):
    if not valid_dts:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(valid_dts, means, "o-", color="tab:blue",  lw=2, ms=7, label="Mean iters")
    ax.plot(valid_dts, maxs,  "s--", color="tab:red", lw=2, ms=7, label="Max iters")

    ax.set_xlabel("Time step dt [s]", fontsize=12)
    ax.set_ylabel("GMRES iterations per solve", fontsize=12)
    ax.set_title(f"{scheme}: GMRES cost vs dt -- P&C Exp 1, dx={dx:.0f} m", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    c_s = 347.0
    dt_cfl = dx / c_s
    ax.axvline(dt_cfl, color="gray", ls=":", lw=1.5,
               label=f"Acoustic CFL=1 dt={dt_cfl:.3f}s")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"pc_gmres_mean_max_vs_dt_{scheme.lower()}_dx{dx:.0f}m.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_wall_time_vs_dt(results_by_dt, scheme, dx):
    """Blown-up dt values are excluded -- see gr_gmres_performance.py for
    why (partial-run wall time isn't comparable, and 'cost per step' would
    divide by the wrong, nominal step count)."""
    dt_vals   = sorted(results_by_dt.keys())
    valid_dts, walls = [], []

    for dt in dt_vals:
        r = results_by_dt[dt]
        if len(r["step_times"]) == 0 or r["blew_up"]:
            continue
        valid_dts.append(dt)
        walls.append(r["wall_time"])

    _draw_wall_time_vs_dt(valid_dts, walls, scheme, dx)


def _draw_wall_time_vs_dt(valid_dts, walls, scheme, dx):
    if not valid_dts:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(valid_dts, walls, "o-", color="tab:green", lw=2, ms=7)
    ax.set_xlabel("dt [s]", fontsize=12)
    ax.set_ylabel("Total wall time [s]", fontsize=12)
    ax.set_title(f"{scheme}: Total wall time vs dt", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cost_per_step = [w / int(round(PAPER["t_end"] / dt))
                     for w, dt in zip(walls, valid_dts)]
    ax.plot(valid_dts, cost_per_step, "o-", color="tab:orange", lw=2, ms=7)
    ax.set_xlabel("dt [s]", fontsize=12)
    ax.set_ylabel("Wall time per step [s]", fontsize=12)
    ax.set_title(f"{scheme}: Cost per step vs dt", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"P&C Exp 1, dx={dx:.0f} m", fontsize=12)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"pc_gmres_wall_time_{scheme.lower()}_dx{dx:.0f}m.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def replot_from_cache(cache_path, scheme, dx):
    """Regenerate Figs 2/3 for one scheme directly from a JSON cache,
    without re-running the (expensive) simulation sweep."""
    import json
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    if scheme not in cache:
        print(f"  '{scheme}' not found in {cache_path} -- skipping")
        return

    dt_vals = sorted(float(dt_str) for dt_str in cache[scheme])

    valid_dts_mm, means, maxs = [], [], []
    valid_dts_wt, walls       = [], []
    for dt in dt_vals:
        r = cache[scheme][str(dt)]
        if r.get("blew_up") or r.get("n_steps", 0) == 0:
            continue
        if r.get("gmres_mean") is not None:
            valid_dts_mm.append(dt)
            means.append(r["gmres_mean"])
            maxs.append(r["gmres_max"])
        valid_dts_wt.append(dt)
        walls.append(r["wall_time"])

    _draw_mean_max_vs_dt(valid_dts_mm, means, maxs, scheme, dx)
    _draw_wall_time_vs_dt(valid_dts_wt, walls, scheme, dx)


def _run_scheme_worker(payload):
    """Module-level (picklable) wrapper so each (scheme, dt) pair runs in
    its own process via ProcessPoolExecutor -- every job is a fully
    independent simulation (fresh grid, fresh state), same pattern as
    pc_si_diffusion_comparison.py's _run_variant_worker."""
    scheme, dx, dt, t_end, kappa = payload
    r = run_scheme(scheme, dx, dt, t_end, kappa=kappa, verbose=True)
    return scheme, dt, r


def main():
    parser = argparse.ArgumentParser(description="GMRES performance analysis for P&C SI/SI2")
    parser.add_argument("--scheme", choices=["SI", "SI2", "SI2LU", "both"], default="SI",
                        help="Which scheme to test (default: SI). SI2LU is SI2's exact "
                             "math solved via a one-time sparse LU factorization instead "
                             "of GMRES -- always reports 0 'gmres_iters' (direct solve). "
                             "Not included in --scheme both.")
    parser.add_argument("--dx", type=float, default=20.0,
                        help="Grid spacing [m] (default: 20, paper resolution)")
    parser.add_argument("--kappa", type=float, default=None,
                        help="Biharmonic (order-4) diffusion coeff [m^4/s] "
                             "(default: damping-timescale value at dx=20m, "
                             "~33 m^4/s -- see pc_si_diffusion_comparison.py). "
                             "Set 0 to disable.")
    parser.add_argument("--plot-from-cache", action="store_true",
                        help="Regenerate Figs 2/3 from an existing "
                             "output/results/pc_gmres_perf_dx{dx}m.json cache "
                             "instead of re-running the simulation sweep.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: one per "
                             "(scheme, dt) job, capped at CPU count)")
    parser.add_argument("--dt-list", type=str, default=None,
                        help="Comma-separated dt values [s] to sweep, overriding the "
                             "built-in default list for every scheme selected via "
                             "--scheme.")
    args = parser.parse_args()

    if args.kappa is None:
        # Same damping-timescale reference as pc_si_diffusion_comparison.py
        kappa_default = 1.0 / (50.0 * (math.pi / 20.0) ** 4)
    else:
        kappa_default = args.kappa

    if args.plot_from_cache:
        cache_path = os.path.join("output", "results", f"pc_gmres_perf_dx{int(args.dx)}m.json")
        if not os.path.exists(cache_path):
            print(f"  No cache found at {cache_path} -- run without --plot-from-cache first.")
            return
        schemes_to_plot = ["SI", "SI2"] if args.scheme == "both" else [args.scheme]
        for sch in schemes_to_plot:
            replot_from_cache(cache_path, sch, args.dx)
        return

    # dt values to sweep, centred on P&C's paper-matching SI/SI2 default of
    # 5s (unlike G&R's 1s default -- P&C's larger bubble scale and coarser
    # dx=20m grid shift the useful dt range up by ~5x).
    dt_list_SI    = [1.25, 2.5, 5.0, 10.0, 20.0, 40.0]
    dt_list_SI2   = [1.25, 2.5, 5.0, 10.0, 20.0]
    dt_list_SI2LU = [1.25, 2.5, 5.0, 10.0, 20.0]

    if args.dt_list is not None:
        override = [float(x) for x in args.dt_list.split(",") if x.strip()]
        dt_list_SI    = override
        dt_list_SI2   = override
        dt_list_SI2LU = override

    schemes = []
    if args.scheme in ("SI", "both"):
        schemes.append(("SI",  dt_list_SI))
    if args.scheme in ("SI2", "both"):
        schemes.append(("SI2", dt_list_SI2))
    if args.scheme == "SI2LU":
        schemes.append(("SI2LU", dt_list_SI2LU))

    jobs = [(sch, dt) for sch, dt_list in schemes for dt in dt_list]
    n_workers = args.workers or min(len(jobs), os.cpu_count() or 1)

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    for sch, dt_list in schemes:
        print(f"\n{'='*60}")
        print(f"  Scheme: {sch}  dx={args.dx:.0f}m  kappa<={kappa_default:.1f} m^4/s "
              f"(auto-capped per dt for explicit-diffusion stability)")
        print(f"  dt values: {dt_list}")
        print(f"{'='*60}")

    print(f"\n  Running {len(jobs)} (scheme, dt) job(s) across {n_workers} "
          f"worker process(es) in parallel...\n")

    all_results = {sch: {} for sch, _ in schemes}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_scheme_worker,
                            (sch, args.dx, dt, PAPER["t_end"], kappa_default)): (sch, dt)
            for sch, dt in jobs
        }
        for fut in as_completed(futures):
            sch, dt, r = fut.result()
            all_results[sch][dt] = r

    for sch, dt_list in schemes:
        results = all_results[sch]
        plot_iters_vs_time(results, sch, args.dx)
        plot_mean_max_vs_dt(results, sch, args.dx)
        plot_wall_time_vs_dt(results, sch, args.dx)

    print(f"\n{'='*60}")
    print("  GMRES PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    for sch, res_by_dt in all_results.items():
        print(f"\n  Scheme {sch}:")
        print(f"  {'dt':>6}  {'steps':>6}  {'mean':>6}  {'max':>6}  {'wall':>8}  {'blowup':>7}")
        for dt in sorted(res_by_dt.keys()):
            r = res_by_dt[dt]
            n_steps = len(r["step_times"])
            if n_steps > 0:
                mn = r["gmres_iters"].mean()
                mx = r["gmres_iters"].max()
            else:
                mn = mx = float("nan")
            print(f"  {dt:6.2f}  {n_steps:6d}  {mn:6.1f}  {mx:6.0f}  "
                  f"{r['wall_time']:8.1f}s  {'YES' if r['blew_up'] else 'no':>7}")

    print(f"\nOutput figures in: {OUT_DIR}/")

    cache = {}
    for sch, res_by_dt in all_results.items():
        cache[sch] = {
            str(dt): {
                "wall_time": r["wall_time"],
                "theta_max": r["theta_max"],
                "w_max":     r["w_max"],
                "blew_up":   bool(r["blew_up"]),
                "n_steps":   int(len(r["step_times"])),
                "gmres_mean": float(r["gmres_iters"].mean()) if len(r["gmres_iters"]) else None,
                "gmres_max":  float(r["gmres_iters"].max())  if len(r["gmres_iters"]) else None,
                "kappa_used": r.get("kappa_used"),
            }
            for dt, r in res_by_dt.items()
        }
    cache_path = os.path.join("output", "results", f"pc_gmres_perf_dx{int(args.dx)}m.json")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    import json
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"Cached dt/wall-time/accuracy summary -> {cache_path}")


if __name__ == "__main__":
    main()
