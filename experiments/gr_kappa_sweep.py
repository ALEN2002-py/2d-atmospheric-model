"""
gr_kappa_sweep.py
=================
G&R Case 2: Rising Thermal Bubble — κ sweep for each diffusion type.

Produces 4 separate vertical profile plots (one per diffusion family):
  1. nabla2  — κ₂ = 0.1, 0.3, 1.0 m²/s
  2. nabla4  — κ₄ = 20, 60, 200 m⁴/s
  3. nabla8  — κ₈ = 2e5, 6e5, 2e6 m⁸/s
  4. Shapiro — period = 30, 60, 120 s

Each plot overlays the IDEAL (no diffusion) and IDEAL tiny-dt reference,
plus 3 lines for the three coefficient values, showing how tuning affects
the vertical profile of θ′ at x = 500 m, t = 700 s.

USAGE
-----
  python experiments/gr_kappa_sweep.py --dx 10
  python experiments/gr_kappa_sweep.py --dx 10 --parallel
"""

import argparse
import os
import sys
import time as wall_time
import types

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------
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
from integrators import step, shapiro_filter

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# G&R Case 2 parameters
# ---------------------------------------------------------------------------
LX, LZ  = 1000.0, 1000.0
THETA_C  = 0.5
R_C      = 250.0
X_C      = 500.0
Z_C      = 350.0
T_END    = 700.0
CS       = 347.0
CFL      = 0.34

# ---------------------------------------------------------------------------
# κ values to sweep (three per family)
# ---------------------------------------------------------------------------
KAPPA2_VALS = [0.1,  0.3,  1.0]    # m²/s  — ×10, ×3, ×1 reduction from original
KAPPA4_VALS = [20.0, 60.0, 200.0]  # m⁴/s
KAPPA8_VALS = [2e5,  6e5,  2e6]    # m⁸/s
SHAPIRO_PERIODS = [30.0, 60.0, 120.0]  # seconds

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
def make_ic(grid):
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X_C)**2 + (grid.z_2d - Z_C)**2)
    state["theta"] = np.where(
        r <= R_C,
        0.5 * THETA_C * (1.0 + np.cos(np.pi * r / R_C)),
        0.0
    )
    return state

# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------
def run_variant(label, dx, dt, grid_params, shapiro_period=None):
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx, **grid_params}
    grid   = Grid(params)
    state  = make_ic(grid)

    use_shapiro = grid_params.pop("_shapiro", False)
    shapiro_every = (max(1, int(round(shapiro_period / dt)))
                     if shapiro_period and shapiro_period > dt else 1)

    nstep   = int(round(T_END / dt))
    dt_exact = T_END / nstep

    t0 = wall_time.perf_counter()
    for n in range(nstep):
        state_new, state, _ = step(state, grid, dt_exact, scheme="RK4")
        state = state_new
        if use_shapiro and (n + 1) % shapiro_every == 0:
            state = shapiro_filter(state, grid)

    elapsed = wall_time.perf_counter() - t0
    th  = state["theta"]
    print(f"  {label:<45}  θ′_max={np.max(th):.3f} K  wall={elapsed:.1f}s",
          flush=True)
    return th.copy()

# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _worker(args):
    label, dx, dt, gp, sp = args
    th = run_variant(label, dx, dt, dict(gp), shapiro_period=sp)
    return label, th

# ---------------------------------------------------------------------------
# Vertical profile plot — one diffusion family
# ---------------------------------------------------------------------------
def plot_profile(title, dx, lines, out_path):
    """
    lines: list of (label, theta_field, color, linestyle, linewidth)
    """
    nz     = int(round(LZ / dx))
    z_km   = (np.arange(nz) + 0.5) * dx / 1000.0
    nx     = int(round(LX / dx))
    x_cen  = (np.arange(nx) + 0.5) * dx
    ix     = int(np.argmin(np.abs(x_cen - X_C)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for label, th, color, ls, lw in lines:
        profile = th[:, ix]
        ax.plot(profile, z_km, label=label,
                color=color, ls=ls, lw=lw, zorder=3)

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z [km]",          fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.02, 0.70)
    ax.set_ylim(0.45, 1.02)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.grid(axis="x", color="#ddd", lw=0.6)
    ax.grid(axis="y", color="#ddd", lw=0.6)
    ax.legend(fontsize=9.5, loc="lower right",
              framealpha=0.92, edgecolor="#ccc", handlelength=2.8)
    fig.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx",       type=float, default=10.0)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    dx      = args.dx
    dt      = CFL * dx / CS
    dt_tiny = dt / 5.0

    print(f"\n{'='*60}")
    print(f"  G&R Case 2 — κ sweep,  dx={dx} m,  dt={dt:.4f} s")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build full variant list
    # ------------------------------------------------------------------
    variants = []

    # References
    variants.append(("IDEAL  (no diffusion)",  dt,      {}, None))
    variants.append(("IDEAL  tiny dt (ref)",   dt_tiny, {}, None))

    # nabla2 sweep
    for k2 in KAPPA2_VALS:
        variants.append((f"nabla2  κ₂={k2} m²/s", dt,
                         {"diffusion_coeff": k2, "diffusion_order": 2}, None))

    # nabla4 sweep
    for k4 in KAPPA4_VALS:
        variants.append((f"nabla4  κ₄={k4} m⁴/s", dt,
                         {"diffusion_coeff": k4, "diffusion_order": 4}, None))

    # nabla8 sweep
    for k8 in KAPPA8_VALS:
        variants.append((f"nabla8  κ₈={k8:.0e} m⁸/s", dt,
                         {"diffusion_coeff": k8, "diffusion_order": 8}, None))

    # Shapiro sweep
    for sp in SHAPIRO_PERIODS:
        variants.append((f"Shapiro  every {sp:.0f} s", dt,
                         {"_shapiro": True}, sp))

    worker_args = [(lbl, dx, vdt, dict(gp), sp)
                   for lbl, vdt, gp, sp in variants]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if args.parallel:
        import multiprocessing
        n_cores = min(len(variants), multiprocessing.cpu_count())
        print(f"  Spawning {n_cores} workers ...\n", flush=True)
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.map(_worker, worker_args)
    else:
        results = [_worker(a) for a in worker_args]

    fields = {lbl: th for lbl, th in results}

    # ------------------------------------------------------------------
    # Reference fields
    # ------------------------------------------------------------------
    ref_ideal    = fields["IDEAL  (no diffusion)"]
    ref_tiny     = fields["IDEAL  tiny dt (ref)"]

    REF_LINES = [
        ("IDEAL  (no diffusion)", ref_ideal, "#1a1a1a", "-",            2.2),
        ("IDEAL  tiny dt (ref)",  ref_tiny,  "#17becf", (0, (6, 2)),    1.8),
    ]

    # Colour ramps per family (light → dark = small → large κ)
    C2 = ["#fc8d59", "#d7191c", "#7f0000"]   # orange → red → dark red
    C4 = ["#a6d96a", "#1a9641", "#004d00"]   # light green → dark green
    C8 = ["#abd9e9", "#2c7bb6", "#08306b"]   # light blue → dark blue
    C_SH = ["#d7b5d8", "#980043", "#49006a"]  # lavender → purple → dark purple

    # ------------------------------------------------------------------
    # Plot 1 — nabla2
    # ------------------------------------------------------------------
    lines = REF_LINES + [
        (f"nabla2  κ₂={k2} m²/s",
         fields[f"nabla2  κ₂={k2} m²/s"], col, "--", 1.6)
        for k2, col in zip(KAPPA2_VALS, C2)
    ]
    plot_profile(
        f"nabla² diffusion — κ₂ sweep\n"
        f"G&R Case 2, dx={dx} m,  t=700 s,  x=500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_gr_nabla2_dx{int(dx)}m.png")
    )

    # ------------------------------------------------------------------
    # Plot 2 — nabla4
    # ------------------------------------------------------------------
    lines = REF_LINES + [
        (f"nabla4  κ₄={k4} m⁴/s",
         fields[f"nabla4  κ₄={k4} m⁴/s"], col, "-.", 1.6)
        for k4, col in zip(KAPPA4_VALS, C4)
    ]
    plot_profile(
        f"nabla⁴ diffusion — κ₄ sweep\n"
        f"G&R Case 2, dx={dx} m,  t=700 s,  x=500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_gr_nabla4_dx{int(dx)}m.png")
    )

    # ------------------------------------------------------------------
    # Plot 3 — nabla8
    # ------------------------------------------------------------------
    lines = REF_LINES + [
        (f"nabla8  κ₈={k8:.0e} m⁸/s",
         fields[f"nabla8  κ₈={k8:.0e} m⁸/s"], col, ":", 1.8)
        for k8, col in zip(KAPPA8_VALS, C8)
    ]
    plot_profile(
        f"nabla⁸ diffusion — κ₈ sweep\n"
        f"G&R Case 2, dx={dx} m,  t=700 s,  x=500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_gr_nabla8_dx{int(dx)}m.png")
    )

    # ------------------------------------------------------------------
    # Plot 4 — Shapiro
    # ------------------------------------------------------------------
    lines = REF_LINES + [
        (f"Shapiro  every {sp:.0f} s",
         fields[f"Shapiro  every {sp:.0f} s"], col, (0, (4, 1.5)), 1.6)
        for sp, col in zip(SHAPIRO_PERIODS, C_SH)
    ]
    plot_profile(
        f"Shapiro (1-2-1) filter — period sweep\n"
        f"G&R Case 2, dx={dx} m,  t=700 s,  x=500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_gr_shapiro_dx{int(dx)}m.png")
    )

    print("\n  Done. 4 vertical profile plots saved to output/figures/\n")


if __name__ == "__main__":
    main()
