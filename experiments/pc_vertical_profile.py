"""
pc_vertical_profile.py
======================
P&C (2022) Exp 1: Convective Bubble — vertical profile of theta' at x=2500m.

Mirrors gr_vertical_profile.py but for the 5km x 5km P&C domain.

Runs 6 variants (IDEAL, nabla2, nabla4, nabla8, Shapiro, IDEAL tiny-dt)
and overlays them on a single vertical profile plot.  Saves:

  output/figures/diffcomp_pc_vprofile_dx{DX}m.png

USAGE
-----
  python experiments/pc_vertical_profile.py             # dx=20m, t=900s  (~15 min)
  python experiments/pc_vertical_profile.py --dx 40     # dx=40m, t=900s  (~3 min)
  python experiments/pc_vertical_profile.py --t-end 1800  # full 30-min run

Reference kappa values (from pc_diffusion_comparison.py):
  dx=10m ref:  kappa2=0.167 m^2/s, kappa4=16.7 m^4/s, kappa8=3.33e5 m^8/s
  Scaled to actual dx inside _kappa()
"""

import argparse
import os
import sys
import time as wall_time
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Dynamic module loading (same pattern as all other experiments)
# ---------------------------------------------------------------------------
def _load_src(name, path):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), os.path.abspath(path), "exec"), mod.__dict__)
    return mod

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_load_src("grid",        os.path.join(_SRC, "grid.py"))
_load_src("dynamics",    os.path.join(_SRC, "dynamics.py"))
_load_src("integrators", os.path.join(_SRC, "integrators.py"))

from grid        import Grid
from integrators import step, shapiro_filter

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# P&C Exp 1 parameters (Pudykiewicz & Clancy 2022, Section 6.1)
# ---------------------------------------------------------------------------
LX, LZ   = 5000.0, 5000.0   # domain [m]
AT        = 0.5              # bubble amplitude [K]
A_CORE    = 400.0            # cylinder radius (20 * dx_ref = 20 * 20m)
SIGMA     = 100.0            # Gaussian tail width (5 * dx_ref)
X_C       = 2500.0           # bubble centre x [m]
Z_C       = 700.0            # bubble centre z [m]
X_PROF    = 2500.0           # vertical profile x-location [m]
CS        = 347.0            # speed of sound [m/s]
CFL       = 0.34             # acoustic CFL for RK4

# Reference kappa values at dx=10m (from pc_diffusion_comparison.py):
KAPPA2_REF = 0.167           # m^2/s
KAPPA4_REF = 16.7            # m^4/s
KAPPA8_REF = 3.33e5          # m^8/s

def _kappa(order, dx, dt):
    """Scale kappa with dx; cap at RK4 explicit stability limit."""
    base = {2: KAPPA2_REF, 4: KAPPA4_REF, 8: KAPPA8_REF}[order]
    raw  = base * (dx / 10.0) ** order
    n    = order // 2
    kappa_max = 0.7 * 2.79 * dx**order / (8**n * dt)
    return min(raw, kappa_max)

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
# Run one variant to t_end
# ---------------------------------------------------------------------------
def run_variant(label, dx, dt, t_end, grid_params, shapiro_period=30.0):
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx}
    params.update({k: v for k, v in grid_params.items() if k != "_shapiro"})
    use_shapiro = grid_params.get("_shapiro", False)
    shapiro_every = (max(1, int(round(shapiro_period / dt)))
                     if shapiro_period and shapiro_period > dt else 1)

    grid  = Grid(params)
    state = make_ic(grid)

    nstep    = int(round(t_end / dt))
    dt_exact = t_end / nstep

    t0 = wall_time.perf_counter()
    for n in range(nstep):
        state_new, _, _ = step(state, grid, dt_exact, scheme="RK4")
        state = state_new
        if use_shapiro and (n + 1) % shapiro_every == 0:
            state = shapiro_filter(state, grid)

    elapsed = wall_time.perf_counter() - t0
    th = state["theta"]
    print(f"  {label:<45}  theta_max={th.max():.4f} K  wall={elapsed:.1f}s",
          flush=True)
    return th.copy(), grid

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_profile(profiles, variant_labels, dx, t_end):
    STYLES = [
        {"color": "#111111", "lw": 2.2, "ls": "-",          "zorder": 10},  # IDEAL
        {"color": "#d62728", "lw": 1.6, "ls": "--",         "zorder":  6},  # nabla2
        {"color": "#ff7f0e", "lw": 1.6, "ls": "-.",         "zorder":  7},  # nabla4
        {"color": "#2ca02c", "lw": 1.6, "ls": ":",          "zorder":  8},  # nabla8
        {"color": "#9467bd", "lw": 1.6, "ls": (0,(4,1.5)),  "zorder":  9},  # Shapiro
        {"color": "#17becf", "lw": 1.4, "ls": "-",          "zorder":  5},  # tiny-dt ref
    ]

    fig, ax = plt.subplots(figsize=(5.5, 6.5))

    for i, ((profile, z_km), label) in enumerate(zip(profiles, variant_labels)):
        s = STYLES[i % len(STYLES)]
        ax.plot(profile, z_km, label=label,
                color=s["color"], lw=s["lw"], ls=s["ls"], zorder=s["zorder"])

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z  [km]",        fontsize=12)
    ax.set_title(
        rf"Vertical profile of $\theta'$ at $x = 2500$ m,  $t = {int(t_end)}$ s"
        + f"\nP&C Exp 1  (dx = {dx} m)",
        fontsize=11)

    ax.set_xlim(left=-0.02)
    ax.set_ylim(0.0, LZ / 1000.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92,
              edgecolor="#cccccc", handlelength=2.8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"diffcomp_pc_vprofile_dx{int(dx)}m.png")
    plt.savefig(out, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  Saved: {out}", flush=True)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P&C Exp 1 — vertical profile of theta' at x=2500m")
    parser.add_argument("--dx",     type=float, default=20.0,
                        help="Grid spacing in m (default 20)")
    parser.add_argument("--t-end",  type=float, default=900.0,
                        help="Simulation end time in s (default 900)")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip tiny-dt reference run (saves ~1 run-time)")
    parser.add_argument("--shapiro-period", type=float, default=30.0,
                        help="Shapiro filter interval in seconds (default 30)")
    args = parser.parse_args()

    dx     = args.dx
    t_end  = args.t_end
    dt     = CFL * dx / CS
    dt_tiny = dt / 5.0
    sp     = args.shapiro_period if args.shapiro_period > 0 else None

    k2 = _kappa(2, dx, dt)
    k4 = _kappa(4, dx, dt)
    k8 = _kappa(8, dx, dt)

    print(f"\n{'='*62}", flush=True)
    print(f"  P&C Exp 1 — vertical profile  dx={dx} m,  t_end={t_end} s", flush=True)
    print(f"  dt={dt:.5f} s   nstep={int(round(t_end/dt))}", flush=True)
    print(f"  kappa2={k2:.4f}  kappa4={k4:.2f}  kappa8={k8:.3e}", flush=True)
    print(f"  Shapiro every {sp if sp else 'every step'} s", flush=True)
    print(f"  Profile at x={X_PROF} m", flush=True)
    print(f"{'='*62}\n", flush=True)

    variants = [
        ("IDEAL  (no diffusion)",          dt,      {}),
        (f"nabla2  (k2={k2:.2g} m2/s)",   dt,      {"diffusion_coeff": k2, "diffusion_order": 2}),
        (f"nabla4  (k4={k4:.2g} m4/s)",   dt,      {"diffusion_coeff": k4, "diffusion_order": 4}),
        (f"nabla8  (k8={k8:.2e} m8/s)",   dt,      {"diffusion_coeff": k8, "diffusion_order": 8}),
        (f"Shapiro (every {sp if sp else 'step'} s)", dt, {"_shapiro": True}),
    ]
    if not args.no_ref:
        variants.append(("IDEAL  tiny dt (ref)", dt_tiny, {}))

    profiles = []
    for label, vdt, gp in variants:
        print(f"Running: {label}", flush=True)
        th, grid = run_variant(label, dx, vdt, t_end, dict(gp), shapiro_period=sp)
        nx_loc   = int(round(LX / dx))
        x_ctrs   = (np.arange(nx_loc) + 0.5) * dx
        ix       = int(np.argmin(np.abs(x_ctrs - X_PROF)))
        nz_loc   = int(round(LZ / dx))
        z_km     = (np.arange(nz_loc) + 0.5) * dx / 1000.0
        profiles.append((th[:, ix], z_km))

    plot_profile(profiles, [v[0] for v in variants], dx, t_end)


if __name__ == "__main__":
    main()
