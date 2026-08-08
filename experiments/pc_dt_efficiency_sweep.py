"""
pc_dt_efficiency_sweep.py
===========================
P&C (2022) Experiment 1: the real (measured, not hardcoded) efficiency
comparison across RK4, SI, and SI2 -- dt vs wall-time vs accuracy, at the
paper's dx=20m resolution.

P&C analogue of gr_dt_efficiency_sweep.py. One difference from the G&R
version: G&R (2008) Table 3 gives a published reference theta'_max to
compare against; P&C (2022) doesn't report an equivalent quantitative
number for Experiment 1 (Figs 1-2 are qualitative). So, following the same
convention pc_diffusion_comparison.py/pc_kappa_sweep.py already use ("IDEAL
tiny dt" as ground truth), the reference here is our OWN finest-dt RK4 run,
not a paper table value.

Blown-up runs are excluded from the accuracy comparison -- their theta_max
reflects a diverging solution, not truncation error.

USAGE
-----
  python experiments/pc_dt_efficiency_sweep.py
  python experiments/pc_dt_efficiency_sweep.py --skip-rk4   # reuse cached RK4 only
"""

import argparse
import json
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
GMRES_CACHE  = os.path.join(RESULTS_DIR, "pc_gmres_perf_dx20m.json")
RK4_CACHE    = os.path.join(RESULTS_DIR, "pc_rk4_dt_sweep_dx20m.json")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# P&C Exp 1 parameters (Section 6.1) -- must match pc_gmres_performance.py's
# t_end=900s so RK4 and SI/SI2 are compared at the same simulated duration.
# ---------------------------------------------------------------------------
LX, LZ = 5000.0, 5000.0
AT     = 0.5
A_RAD  = 400.0
SIGMA  = 100.0
X0, Z0 = 2500.0, 700.0
T_END  = 900.0
DX     = 20.0
CS     = 347.0

RK4_DT_VALS = None   # computed in main() from CFL, since it depends on DX


def _kappa_used(dt, dx=20.0, kappa_requested=None):
    """
    Recompute the auto-capped order-4 diffusion coefficient actually used by
    pc_gmres_performance.py for a given dt (same formula as that script's
    run_scheme()). Fallback for cache entries predating 'kappa_used'.
    """
    global _kappa_fallback_warned
    if kappa_requested is None:
        kappa_requested = 1.0 / (50.0 * (math.pi / 20.0) ** 4)   # pc_gmres_performance.py default
    if not _kappa_fallback_warned:
        print(f"  WARNING: one or more cache entries are missing 'kappa_used' -- "
              f"assuming --kappa={kappa_requested:.3f} (pc_gmres_performance.py's "
              f"default) for all of them. Re-run that script to refresh the "
              f"cache with the real value(s) if it was ever run with a "
              f"different --kappa.")
        _kappa_fallback_warned = True
    eig_max_o4   = (8.0 / dx**2) ** 2
    kappa_stable = 0.7 * 2.0 / (eig_max_o4 * dt) if dt > 0 else kappa_requested
    return min(kappa_requested, kappa_stable)

_kappa_fallback_warned = False


def make_ic(grid):
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X0)**2 + (grid.z_2d - Z0)**2)
    state["theta"] = np.where(
        r <= A_RAD, AT, AT * np.exp(-(r - A_RAD)**2 / (2.0 * SIGMA**2)),
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
    process via ProcessPoolExecutor -- the two dt points are fully
    independent, same pattern as pc_si_diffusion_comparison.py."""
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

    dt_base = 0.34 * DX / CS   # matches pc_diffusion_comparison.py's CFL convention
    dt_vals = [dt_base, dt_base / 2.0]   # smaller (2nd) point is our ground truth

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    print(f"  Running {len(dt_vals)} RK4 dt value(s) in parallel "
          f"(this will take a while at dx={DX:.0f}m)...", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=len(dt_vals)) as executor:
        futures = {executor.submit(_run_rk4_worker, dt): dt for dt in dt_vals}
        for fut in as_completed(futures):
            dt, r = fut.result()
            results[str(round(dt, 6))] = r
            print(f"    dt={dt:.4f}s: theta_max={r['theta_max']:.3f}K  "
                  f"w_max={r['w_max']:.3f}m/s  wall={r['wall_time']:.1f}s", flush=True)

    with open(RK4_CACHE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Cached RK4 results -> {RK4_CACHE}")
    return results


def load_gmres_cache():
    if not os.path.exists(GMRES_CACHE):
        raise FileNotFoundError(
            f"{GMRES_CACHE} not found -- run "
            f"'python experiments/pc_gmres_performance.py --scheme both' first."
        )
    with open(GMRES_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def _reference_theta(rk4_results):
    """Ground truth = our own finest-dt RK4 run (smallest dt key), since P&C
    (2022) doesn't publish a quantitative theta'_max for Experiment 1 the
    way G&R (2008) Table 3 does for Case 2."""
    valid = {k: v for k, v in rk4_results.items() if not v.get("blew_up")}
    if not valid:
        return None
    finest_dt = min(valid.keys(), key=lambda k: float(k))
    return valid[finest_dt]["theta_max"], finest_dt


def make_frontier_plot(rk4_results, gmres_results, theta_ref):
    fig, ax = plt.subplots(figsize=(9, 6))

    styles = {
        "RK4":   {"color": "#d62728", "marker": "o"},
        "SI":    {"color": "#9467bd", "marker": "s"},
        "SI2":   {"color": "#2ca02c", "marker": "^"},
        "SI2LU": {"color": "#1f77b4", "marker": "D"},
    }

    for dt_str, r in rk4_results.items():
        if r.get("blew_up"):
            continue
        err = abs(r["theta_max"] - theta_ref) / theta_ref * 100.0
        s = styles["RK4"]
        ax.scatter([r["wall_time"]], [err], s=110, color=s["color"],
                   marker=s["marker"], zorder=5,
                   label="RK4 (measured)" if dt_str == list(rk4_results)[0] else None)
        ax.annotate(f"RK4 dt={float(dt_str):.4f}s", xy=(r["wall_time"], err),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=s["color"])

    for scheme in ("SI", "SI2", "SI2LU"):
        first = True
        for dt_str, r in gmres_results.get(scheme, {}).items():
            if r.get("blew_up"):
                continue
            err = abs(r["theta_max"] - theta_ref) / theta_ref * 100.0
            kappa = r.get("kappa_used")
            if kappa is None:
                kappa = _kappa_used(float(dt_str))
            s = styles[scheme]
            ax.scatter([r["wall_time"]], [err], s=110, color=s["color"],
                       marker=s["marker"], zorder=5,
                       label=f"{scheme} (measured)" if first else None)
            ax.annotate(f"{scheme} dt={dt_str}s\n" + r"$\kappa_4$=" + f"{kappa:.2f}",
                        xy=(r["wall_time"], err),
                        xytext=(6, -22 if first else 6), textcoords="offset points",
                        fontsize=7.5, color=s["color"])
            first = False

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock time [s]  (log scale)", fontsize=12)
    ax.set_ylabel(r"Error in $\theta'_{\max}$ relative to finest-dt RK4 [%]", fontsize=11)
    ax.set_title(
        f"Measured Efficiency Frontier -- P&C Exp 1 (dx={DX:.0f} m, t_end={T_END:.0f} s)\n"
        "All points measured directly. Reference = finest-dt RK4 (no published "
        "P&C theta'_max exists). Blown-up runs excluded.",
        fontsize=10.5,
    )
    ax.axhline(y=0, color="#aaa", lw=0.7, ls="--")
    ax.grid(True, which="both", color="#e8e8e8", lw=0.5)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#ccc")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "pc_dt_efficiency_frontier_measured.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_summary_table(rk4_results, gmres_results, theta_ref, ref_dt):
    print("\n" + "=" * 100)
    print(f"  MEASURED EFFICIENCY COMPARISON -- P&C Exp 1, dx={DX:.0f} m, t_end={T_END:.0f} s")
    print("=" * 100)
    print(f"  Reference: our own finest-dt RK4 run (dt={ref_dt}s) -- "
          f"theta'_max = {theta_ref:.4f} K")
    print(f"  {'Scheme':<6} {'dt(s)':>8} {'wall(s)':>10} {'theta_max(K)':>13} "
          f"{'err_vs_ref(%)':>14} {'kappa4_used':>12} {'blowup':>8}")
    print("  " + "-" * 96)

    def _row(scheme, dt_str, r, kappa=None):
        blow = "YES" if r.get("blew_up") else "no"
        if r.get("blew_up"):
            err_str = "N/A (diverged)"
        else:
            err = abs(r["theta_max"] - theta_ref) / theta_ref * 100.0
            err_str = f"{err:.1f}"
        kappa_str = f"{kappa:.2f}" if kappa is not None else "0 (none)"
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
    print("=" * 100)
    print("  NOTE: kappa4_used shrinks automatically at larger dt (stability cap) --")
    print("  SI/SI2 accuracy improving with dt partly reflects less damping, not")
    print("  purely improving temporal truncation error. See discussion in text.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real (measured) dt/wall-time/accuracy comparison: RK4 vs SI vs SI2 (P&C)"
    )
    parser.add_argument("--skip-rk4", action="store_true",
                        help="Don't run RK4 fresh; use cache only (omit if no cache)")
    args = parser.parse_args()

    gmres_results = load_gmres_cache()
    rk4_results   = get_rk4_results(args.skip_rk4)

    ref = _reference_theta(rk4_results)
    if ref is None:
        print("  No valid (non-blown-up) RK4 run available -- cannot compute error %. "
              "Run without --skip-rk4 first.")
        return
    theta_ref, ref_dt = ref

    print_summary_table(rk4_results, gmres_results, theta_ref, ref_dt)
    make_frontier_plot(rk4_results, gmres_results, theta_ref)


if __name__ == "__main__":
    main()
