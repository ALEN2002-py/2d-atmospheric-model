"""
gr_vertical_profile.py
======================
Vertical profile of theta' at x=500m, t=700s for all 6 diffusion variants.
Mirrors G&R (2008) Fig. 4.

  python experiments/gr_vertical_profile.py            # dx=20m
  python experiments/gr_vertical_profile.py --dx 40
  python experiments/gr_vertical_profile.py --dx 100   # quick test (<45s)
"""

import argparse
import os
import sys
import time
import types

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Module loader
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
# Parameters
# ---------------------------------------------------------------------------
LX, LZ   = 1000.0, 1000.0   # domain size [m]
THETA_C  = 0.5               # bubble amplitude [K]
R_C      = 250.0             # bubble radius [m]
X_C      = 500.0             # bubble centre x [m]
Z_C      = 350.0             # bubble centre z [m]
T_END    = 700.0             # simulation end time [s]
CS       = 347.0             # speed of sound [m/s]  sqrt(gamma*Rd*T0), T0=300K
CFL      = 0.34              # max acoustic CFL for RK4 stability

# Reference kappa values at dx=10 m — scaled inside _kappa() to actual dx.
# Chosen so the damping timescale at the 2*dx wave is ~50–100 s at dx=10 m.
KAPPA2_REF = 1.0      # m^2/s  (standard Laplacian)
KAPPA4_REF = 200.0    # m^4/s  (biharmonic)
KAPPA8_REF = 2.0e6    # m^8/s  (octaharmonic)

def _kappa(order, dx, dt):
    """Scale kappa with dx; cap at the RK4 explicit stability limit.

    Max eigenvalue of the discrete 2D Laplacian (nabla^2) at Nyquist = 8/dx^2.
    For (nabla^2)^n: eigenvalue magnitude = (8/dx^2)^n.
    RK4 stability: kappa * (8/dx^2)^n * dt <= 2.79.  Safety factor 70%.
    """
    base      = {2: KAPPA2_REF, 4: KAPPA4_REF, 8: KAPPA8_REF}[order]
    raw       = base * (dx / 10.0) ** order
    n         = order // 2
    kappa_max = 0.7 * 2.79 * dx**order / (8**n * dt)
    return min(raw, kappa_max)

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def make_ic(grid):
    """Cosine-bell thermal bubble initial condition (G&R 2008, eq. 3.1).

    theta'(x,z) = (THETA_C/2)*(1+cos(pi*r/R_C))  for r <= R_C
    theta'(x,z) = 0                                for r >  R_C
    """
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X_C)**2 + (grid.z_2d - Z_C)**2)
    state["theta"] = np.where(
        r <= R_C, 0.5 * THETA_C * (1.0 + np.cos(np.pi * r / R_C)), 0.0)
    return state

# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------
def run_to_end(label, dx, dt, grid_params, shapiro_period=30.0):
    """
    Run G&R Case 2 to T_END with the given grid_params.

    Parameters
    ----------
    label         : str   — variant name (for progress output)
    dx            : float — grid spacing [m]
    dt            : float — time step [s]
    grid_params   : dict  — extra Grid() kwargs; '_shapiro' key enables the filter
    shapiro_period: float — apply Shapiro filter every this many seconds (default 30)

    Returns
    -------
    theta_field : np.ndarray shape (nz, nx) — theta' at T_END
    grid        : Grid
    """
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx}
    params.update({k: v for k, v in grid_params.items() if k != "_shapiro"})
    use_shapiro    = grid_params.get("_shapiro", False)
    # shapiro_period=None means every step; 30.0 = every 30s; dt = every step explicitly
    if shapiro_period is None or shapiro_period <= dt:
        shapiro_every = 1
    else:
        shapiro_every = max(1, int(round(shapiro_period / dt)))

    grid  = Grid(params)
    state = make_ic(grid)

    nstep = int(round(T_END / dt))
    dt_e  = T_END / nstep

    t0 = time.perf_counter()
    for n in range(nstep):
        state_new, _, _ = step(state, grid, dt_e, scheme="RK4")
        state = state_new
        if use_shapiro and (n + 1) % shapiro_every == 0:
            state = shapiro_filter(state, grid)

    elapsed = time.perf_counter() - t0
    print(f"  {label:<38}  theta_max={state['theta'].max():.3f} K  "
          f"wall={elapsed:.1f}s", flush=True)
    return state["theta"], grid

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_profile(profiles_z, variant_labels, dx):
    STYLES = [
        {"color": "#111111", "lw": 2.2, "ls": "-",         "zorder": 10},
        {"color": "#d62728", "lw": 1.6, "ls": "--",        "zorder":  6},
        {"color": "#ff7f0e", "lw": 1.6, "ls": "-.",        "zorder":  7},
        {"color": "#2ca02c", "lw": 1.6, "ls": ":",         "zorder":  8},
        {"color": "#9467bd", "lw": 1.6, "ls": (0,(4,1.5)), "zorder":  9},
        {"color": "#888888", "lw": 1.4, "ls": "-",         "zorder":  5},
    ]

    fig, ax = plt.subplots(figsize=(5.0, 6.2))

    for i, ((profile, z_km), label) in enumerate(zip(profiles_z, variant_labels)):
        s = STYLES[i % len(STYLES)]
        ax.plot(profile, z_km, label=label,
                color=s["color"], lw=s["lw"], ls=s["ls"], zorder=s["zorder"])

    ax.axvline(0.05, color="#aaaaaa", lw=0.8, ls="--", zorder=1,
               label="G&R min contour (0.05 K)")

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z  [km]", fontsize=12)
    ax.set_title(
        r"Vertical profile of $\theta'$ at $x = 500$ m,  $t = 700$ s"
        + f"\nG&R Case 2  (dx = {dx} m)", fontsize=11)
    ax.set_xlim(left=-0.03)
    ax.set_ylim(0, LZ / 1000)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92,
              edgecolor="#cccccc", handlelength=2.8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"diffcomp_gr_vprofile_dx{int(dx)}m.png")
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
        description="G&R Case 2 vertical profile at x=500m, t=700s")
    parser.add_argument("--dx", type=float, default=20.0)
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip tiny-dt reference run")
    parser.add_argument("--shapiro-period", type=float, default=30.0,
                        help="Shapiro filter interval in simulation seconds "
                             "(default 30; use 0 for every step)")
    args = parser.parse_args()

    dx = args.dx
    dt = CFL * dx / CS
    dt_tiny = dt / 5.0
    sp = args.shapiro_period if args.shapiro_period > 0 else None  # None → every step

    k2 = _kappa(2, dx, dt)
    k4 = _kappa(4, dx, dt)
    k8 = _kappa(8, dx, dt)

    include_ref = (not args.no_ref) and (dx <= 60.0)

    print(f"\n{'='*60}", flush=True)
    print(f"  G&R Case 2 — vertical profile  (dx={dx} m)", flush=True)
    print(f"  dt={dt:.4f} s   nstep={int(round(T_END/dt))}", flush=True)
    print(f"  kappa2={k2:.2f}  kappa4={k4:.1f}  kappa8={k8:.2e}", flush=True)
    print(f"  ref run: {'yes (dt/5)' if include_ref else 'skipped'}", flush=True)
    print(f"{'='*60}\n", flush=True)

    variants = [
        ("IDEAL  (no diffusion)",          dt,      {}),
        (f"nabla2  (k={k2:.2g} m2/s)",    dt,      {"diffusion_coeff": k2, "diffusion_order": 2}),
        (f"nabla4  (k={k4:.2g} m4/s)",    dt,      {"diffusion_coeff": k4, "diffusion_order": 4}),
        (f"nabla8  (k={k8:.2e} m8/s)",    dt,      {"diffusion_coeff": k8, "diffusion_order": 8}),
        (f"Shapiro (every {sp if sp else 'step'} s)", dt, {"_shapiro": True}),
    ]
    if include_ref:
        variants.append(
            ("IDEAL tiny dt (ref)", dt_tiny, {}))

    profiles_z  = []
    for label, vdt, gp in variants:
        print(f"Running: {label}", flush=True)
        theta_field, grid = run_to_end(label, dx, vdt, dict(gp), shapiro_period=sp)
        nx_loc = int(round(LX / dx))
        x_centres = (np.arange(nx_loc) + 0.5) * dx
        ix = int(np.argmin(np.abs(x_centres - X_C)))
        nz_loc = int(round(LZ / dx))
        z_km = (np.arange(nz_loc) + 0.5) * dx / 1000.0
        profiles_z.append((theta_field[:, ix], z_km))

    variant_labels = [v[0] for v in variants]
    plot_profile(profiles_z, variant_labels, dx)


if __name__ == "__main__":
    main()
