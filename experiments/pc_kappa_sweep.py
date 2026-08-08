"""
pc_kappa_sweep.py
=================
P&C Exp 1: Convective Bubble (5 km domain) — κ sweep for each diffusion type.

Produces 4 separate vertical profile plots (one per diffusion family):
  1. nabla2  — κ₂ = 0.05, 0.167, 0.5 m²/s
  2. nabla4  — κ₄ = 11, 33.3, 100 m⁴/s
  3. nabla8  — κ₈ = 1.1e5, 3.33e5, 1e6 m⁸/s
  4. Shapiro — period = 30, 60, 120 s

Each plot overlays the IDEAL (no diffusion) and IDEAL tiny-dt reference,
plus 3 lines for the three coefficient values.

USAGE
-----
  python experiments/pc_kappa_sweep.py --dx 20
  python experiments/pc_kappa_sweep.py --dx 20 --parallel
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
# P&C Exp 1 parameters
# ---------------------------------------------------------------------------
LX, LZ   = 5000.0, 5000.0
AT        = 0.5      # K — bubble amplitude
A_CORE    = 400.0    # m — cylinder radius (20 * dx_ref)
SIGMA     = 100.0    # m — Gaussian tail width (5 * dx_ref)
X_C       = 2500.0   # m — bubble centre x
Z_C       = 700.0    # m — bubble centre z
T_END     = 1200.0   # s   — IDEAL blows up at ~1126 s; κ variants stable
X_PROF    = 2500.0   # m — vertical profile x-location
CS        = 347.0    # m/s — speed of sound
CFL       = 0.34

# ---------------------------------------------------------------------------
# κ sweep values (1/3 ×, 1 ×, 3 × around reference) — dx=20m SCALED values
# Reference at dx=10m: κ₂=0.167, κ₄=16.7, κ₈=3.33e5
# Scaled to dx=20m: κ₂ × 4, κ₄ × 16, κ₈ × 256
#   → κ₂_ref=0.67 m²/s, κ₄_ref=267 m⁴/s, κ₈_ref=8.52e7 m⁸/s
# ---------------------------------------------------------------------------
KAPPA2_VALS     = [0.22,  0.67,   2.0  ]   # m²/s   (1/3×, 1×, 3× of 0.67)
KAPPA4_VALS     = [90.0,  267.0,  800.0]   # m⁴/s   (1/3×, 1×, 3× of 267)
KAPPA8_VALS     = [2.8e7, 8.5e7, 2.5e8]   # m⁸/s   (1/3×, 1×, 3× of 8.52e7)
SHAPIRO_PERIODS = [30.0,  60.0,  120.0  ]   # seconds (unchanged — all stable)

# ---------------------------------------------------------------------------
# Initial condition — cylindrical core + Gaussian edge
# ---------------------------------------------------------------------------
def make_ic(grid):
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X_C)**2 + (grid.z_2d - Z_C)**2)
    theta = np.where(
        r <= A_CORE,
        AT,
        AT * np.exp(-(r - A_CORE)**2 / (2.0 * SIGMA**2))
    )
    theta[theta < 0.01] = 0.0
    state["theta"] = theta
    return state

# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------
def run_variant(label, dx, dt, grid_params, shapiro_period=None):
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx, **grid_params}
    grid   = Grid(params)
    state  = make_ic(grid)

    use_shapiro   = grid_params.pop("_shapiro", False)
    shapiro_every = (max(1, int(round(shapiro_period / dt)))
                     if shapiro_period and shapiro_period > dt else 1)

    nstep    = int(round(T_END / dt))
    dt_exact = T_END / nstep

    blew_up = False
    last_valid_th = None   # last theta field that was finite
    t0 = wall_time.perf_counter()
    for n in range(nstep):
        state_new, state, _ = step(state, grid, dt_exact, scheme="RK4")
        state = state_new
        if use_shapiro and (n + 1) % shapiro_every == 0:
            state = shapiro_filter(state, grid)
        if np.all(np.isfinite(state["w"])):
            last_valid_th = state["theta"].copy()   # checkpoint last good state
        else:
            t_sim = (n + 1) * dt_exact
            print(f"  BLOW-UP at t={t_sim:.1f}s — aborting {label}.", flush=True)
            blew_up = True
            break

    elapsed = wall_time.perf_counter() - t0
    th = state["theta"]
    status = " [BLOW-UP]" if blew_up else ""
    print(f"  {label:<45}  θ′_max={np.nanmax(th):.4f} K  wall={elapsed:.1f}s{status}",
          flush=True)
    # On blow-up: return last valid theta (not zeros) so reference line is meaningful
    if blew_up:
        return last_valid_th if last_valid_th is not None else np.zeros_like(th), True
    return th.copy(), False

# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _worker(args):
    label, dx, dt, gp, sp = args
    th, blew_up = run_variant(label, dx, dt, dict(gp), shapiro_period=sp)
    return label, th, blew_up

# ---------------------------------------------------------------------------
# Vertical profile plot — one diffusion family
# ---------------------------------------------------------------------------
def plot_profile(title, dx, lines, out_path):
    nz    = int(round(LZ / dx))
    z_km  = (np.arange(nz) + 0.5) * dx / 1000.0
    nx    = int(round(LX / dx))
    x_arr = (np.arange(nx) + 0.5) * dx
    ix    = int(np.argmin(np.abs(x_arr - X_PROF)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for label, th, color, ls, lw in lines:
        ax.plot(th[:, ix], z_km, label=label,
                color=color, ls=ls, lw=lw, zorder=3)

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z [km]",          fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.02, 0.70)
    ax.set_ylim(1.5, 3.8)   # bubble cap at z≈3 km at t=1200 s
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
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
    parser.add_argument("--dx",       type=float, default=20.0)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    dx      = args.dx
    dt      = CFL * dx / CS

    print(f"\n{'='*60}")
    print(f"  P&C Exp 1 — κ sweep,  dx={dx} m,  dt={dt:.5f} s")
    print(f"  Grid: {int(LX/dx)} × {int(LZ/dx)},  nsteps ≈ {int(round(T_END/dt))}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Build full variant list
    # ------------------------------------------------------------------
    variants = []

    variants.append(("IDEAL  (no diffusion)",  dt,      {}, None))

    for k2 in KAPPA2_VALS:
        variants.append((f"nabla2  κ₂={k2} m²/s", dt,
                         {"diffusion_coeff": k2, "diffusion_order": 2}, None))

    for k4 in KAPPA4_VALS:
        variants.append((f"nabla4  κ₄={k4} m⁴/s", dt,
                         {"diffusion_coeff": k4, "diffusion_order": 4}, None))

    for k8 in KAPPA8_VALS:
        variants.append((f"nabla8  κ₈={k8:.2e} m⁸/s", dt,
                         {"diffusion_coeff": k8, "diffusion_order": 8}, None))

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

    fields   = {lbl: th        for lbl, th, _ in results}
    blowups  = {lbl: blew_up  for lbl, _,  blew_up in results}

    # ------------------------------------------------------------------
    # Reference fields
    # ------------------------------------------------------------------
    ref_ideal   = fields["IDEAL  (no diffusion)"]
    ideal_label = ("IDEAL  (no diffusion, last valid t≈900 s)"
                   if blowups["IDEAL  (no diffusion)"]
                   else "IDEAL  (no diffusion)")
    ideal_ls    = (0, (3, 2)) if blowups["IDEAL  (no diffusion)"] else "-"

    REF_LINES = [
        (ideal_label, ref_ideal, "#1a1a1a", ideal_ls, 2.2),
    ]

    # Colour ramps per family
    C2   = ["#fc8d59", "#d7191c", "#7f0000"]
    C4   = ["#a6d96a", "#1a9641", "#004d00"]
    C8   = ["#abd9e9", "#2c7bb6", "#08306b"]
    C_SH = ["#d7b5d8", "#980043", "#49006a"]

    # Plot 1 — nabla2
    lines = REF_LINES + [
        (f"nabla2  κ₂={k2} m²/s",
         fields[f"nabla2  κ₂={k2} m²/s"], col, "--", 1.6)
        for k2, col in zip(KAPPA2_VALS, C2)
    ]
    plot_profile(
        f"nabla² diffusion — κ₂ sweep\n"
        f"P&C Exp 1, dx={dx} m,  t=1200 s,  x=2500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_pc_nabla2_dx{int(dx)}m.png")
    )

    # Plot 2 — nabla4
    lines = REF_LINES + [
        (f"nabla4  κ₄={k4} m⁴/s",
         fields[f"nabla4  κ₄={k4} m⁴/s"], col, "-.", 1.6)
        for k4, col in zip(KAPPA4_VALS, C4)
    ]
    plot_profile(
        f"nabla⁴ diffusion — κ₄ sweep\n"
        f"P&C Exp 1, dx={dx} m,  t=1200 s,  x=2500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_pc_nabla4_dx{int(dx)}m.png")
    )

    # Plot 3 — nabla8
    lines = REF_LINES + [
        (f"nabla8  κ₈={k8:.2e} m⁸/s",
         fields[f"nabla8  κ₈={k8:.2e} m⁸/s"], col, ":", 1.8)
        for k8, col in zip(KAPPA8_VALS, C8)
    ]
    plot_profile(
        f"nabla⁸ diffusion — κ₈ sweep\n"
        f"P&C Exp 1, dx={dx} m,  t=1200 s,  x=2500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_pc_nabla8_dx{int(dx)}m.png")
    )

    # Plot 4 — Shapiro
    lines = REF_LINES + [
        (f"Shapiro  every {sp:.0f} s",
         fields[f"Shapiro  every {sp:.0f} s"], col, (0, (4, 1.5)), 1.6)
        for sp, col in zip(SHAPIRO_PERIODS, C_SH)
    ]
    plot_profile(
        f"Shapiro (1-2-1) filter — period sweep\n"
        f"P&C Exp 1, dx={dx} m,  t=1200 s,  x=2500 m",
        dx, lines,
        os.path.join(OUT_DIR, f"kappasweep_pc_shapiro_dx{int(dx)}m.png")
    )

    print("\n  Done. 4 vertical profile plots saved to output/figures/\n")


if __name__ == "__main__":
    main()
