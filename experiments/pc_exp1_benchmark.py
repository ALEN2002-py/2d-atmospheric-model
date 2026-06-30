"""
pc_exp1_benchmark.py
====================
P&C (2022) Experiment 1: Convective Bubble -- exact paper replication.

All parameters taken directly from:
  Pudykiewicz, J.A. and Clancy, C. (2022).
  "Convection experiments with the exponential time integration scheme."
  J. Comput. Phys. 449, 110803.
  DOI: 10.1016/j.jcp.2021.110803

Experiment 1 parameters (Section 6.1)
----------------------------------------
  Domain    : 5000 m x 5000 m  (delta = 20 m)
  Grid      : nx = nz = 250 (unstaggered, 4th-order centred FD in paper)
  Base state: isentropic, theta_bar = 300 K
  Bubble    : theta'(x,z) = AT                           for r <= a
              theta'(x,z) = AT * exp(-(r-a)^2/(2*sigma^2)) for r > a
              AT = 0.5 K,  a = 20*delta = 400 m,  sigma = 5*delta = 100 m
              Centre: (x0, z0) = (2500 m, 700 m)
  dt        : 15 s  (Courant number ~250 in paper)
  t_end     : 1800 s (~30 min; velocity max at ~600 s = 3 x T_buoyancy)
  Filter    : Shapiro filter applied every 2 time steps (ESSENTIAL in paper)
  Viscosity : None (inviscid)

Physical timescales (Section 6.1):
  B0 = g * AT / theta_bar  = 9.81 * 0.5 / 300 ~ 0.01635 m/s^2
  L  = 2a = 800 m
  T  = sqrt(L/B0) ~ 220 s    (buoyancy timescale)
  W  = sqrt(B0*L) ~ 3.6 m/s (velocity scale)
  3T ~ 660 s -- time of maximum velocity in paper

Usage
-----
    cd 2d-atmospheric-model
    python experiments/pc_exp1_benchmark.py --scheme RK4 --dx 40 --t_end 1200
    python experiments/pc_exp1_benchmark.py --scheme EPI3FJ --dx 40 --t_end 600
"""

import argparse
import os
import sys
import time as wall_time
import types


def _load_src(name, path):
    """Load a .py source file as a named module without installing the package.

    Executes the source via compile()+exec() so imports are always
    resolved from the live .py file, bypassing any stale .pyc bytecache.
    """
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
from matplotlib.colors import Normalize

from grid        import Grid
from integrators import step, shapiro_filter

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Paper parameters (Section 6.1)
# ---------------------------------------------------------------------------

PAPER = {
    "Lx":  5000.0,
    "Lz":  5000.0,
    "delta": 20.0,
    "AT":    0.5,
    "a":   400.0,
    "sigma": 100.0,
    "x0":  2500.0,
    "z0":   700.0,
    "t_end": 1800.0,
    "dt_EPI":  15.0,
}

_g    = 9.81
_AT   = PAPER["AT"]
_tbar = 300.0
_B0   = _g * _AT / _tbar
_L    = 2.0 * PAPER["a"]
T_BUOY  = np.sqrt(_L / _B0)
W_SCALE = np.sqrt(_B0 * _L)

SNAP_TIMES = [0, 300, 600, 900, 1200, 1800]


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def make_state(grid):
    """
    Set up P&C (2022) Experiment 1 initial condition (Section 6.1).

    The bubble has a flat-top cylindrical core with a smooth Gaussian edge:
      theta' = AT                              for r <= a   (flat top)
      theta' = AT * exp(-(r-a)^2/(2*sigma^2)) for r >  a   (Gaussian decay)

    This differs from the G&R cosine bell (which tapers to zero at r=r_c).
    The flat-top shape means theta'=AT everywhere inside the core radius a.

    Parameters
    ----------
    grid : Grid  — provides x_2d, z_2d coordinate arrays

    Returns
    -------
    state : dict  — {u, w, theta, pi}; only theta is non-zero
    """
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - PAPER["x0"])**2
              + (grid.z_2d - PAPER["z0"])**2)
    a     = PAPER["a"]
    sigma = PAPER["sigma"]
    AT    = PAPER["AT"]
    state["theta"] = np.where(
        r <= a,
        AT,
        AT * np.exp(-(r - a)**2 / (2.0 * sigma**2)),
    )
    return state


# ---------------------------------------------------------------------------
# Time integration loop
# ---------------------------------------------------------------------------

def run(scheme="RK4", dx=40.0, dt=None, t_end=None, shapiro=True, kappa=0.0):
    if t_end is None:
        t_end = PAPER["t_end"]

    is_epi     = scheme in ("EPI2", "EPI3", "EPI2FJ", "EPI3FJ")
    is_explicit = scheme in ("RK4", "FTCS")

    if dt is None:
        if is_explicit:
            c_s = np.sqrt((1004.0 / 717.0) * 287.0 * _tbar)
            dt  = 0.9 * dx / c_s
        else:
            dt = PAPER["dt_EPI"]

    grid  = Grid({"Lx": PAPER["Lx"], "Lz": PAPER["Lz"], "dx": dx, "dz": dx,
                  "diffusion_coeff": kappa})
    state = make_state(grid)

    n_steps = int(round(t_end / dt))
    c_s     = np.sqrt((grid.cp / grid.cv) * grid.Rd * grid.T0)
    cfl_ac  = c_s * dt / dx

    print("")
    print("="*65)
    print("  P&C (2022) Experiment 1 -- Convective Bubble")
    print("  Scheme : {}".format(scheme))
    print("  Grid   : {}x{}  (dx=dz={:.0f} m)".format(grid.nx, grid.nz, dx))
    print("  dt     : {} s   acoustic CFL = {:.1f}".format(dt, cfl_ac))
    print("  Steps  : {}   t_end = {:.0f} s".format(n_steps, t_end))
    print("  Filter : Shapiro {}".format("ON (every 2 steps)" if shapiro else "OFF"))
    if kappa > 0:
        print("  Diffusion: kappa = {:.1f} m^2/s  (tau_2dx={:.0f}s, tau_bubble={:.0f}s)".format(
            kappa, (2*dx)**2/kappa, PAPER["a"]**2/kappa))
    print("  T_buoy ~ {:.0f} s   ({:.1f} x T)".format(T_BUOY, t_end/T_BUOY))
    print("  W_scale ~ {:.2f} m/s".format(W_SCALE))
    print("="*65)

    snap_set = set(t for t in SNAP_TIMES if t <= t_end)
    snap_set.add(t_end)

    snapshots  = []
    epi_n_prev = None
    state_old  = None
    t          = 0.0
    t0_wall    = wall_time.perf_counter()

    snapshots.append((0.0, {k: v.copy() for k, v in state.items()}))

    for n in range(n_steps):
        try:
            state_new, state_old, epi_extra = step(
                state, grid, dt,
                scheme     = scheme,
                state_old  = state_old,
                epi_n_prev = epi_n_prev,
            )
        except Exception as exc:
            print("\n  Exception at step {} (t={:.1f}s): {}".format(n+1, t, exc))
            break

        state = state_new

        if is_epi and epi_extra is not None:
            if scheme in ("EPI2FJ", "EPI3FJ"):
                epi_n_prev = epi_extra
            else:
                epi_n_prev = epi_extra["n_rhs"]

        # Shapiro filter: every 2 steps for EPI only (as in paper)
        if shapiro and is_epi and (n + 1) % 2 == 0:
            state = shapiro_filter(state, grid)

        t += dt

        if not np.all(np.isfinite(state["w"])):
            print("\n  BLOW-UP at t={:.1f}s -- aborting.".format(t))
            break

        for ts in list(snap_set):
            if abs(t - ts) < 0.5 * dt and ts > 0:
                snapshots.append((t, {k: v.copy() for k, v in state.items()}))
                snap_set.discard(ts)

        report_every = max(1, int(round(300.0 / dt)))
        if (n + 1) % report_every == 0:
            vmax = np.sqrt(state["u"]**2 + state["w"]**2).max()
            print("  t={:7.1f}s  |theta'|max={:.4f} K  |v|max={:.4f} m/s".format(
                t, np.max(np.abs(state["theta"])), vmax))

    elapsed = wall_time.perf_counter() - t0_wall
    print("\n  Run complete: {:.1f} s wall time  ({} snapshots)".format(
        elapsed, len(snapshots)))

    diag = {
        "t":         t,
        "theta_max": float(np.max(state["theta"])),
        "theta_min": float(np.min(state["theta"])),
        "w_max":     float(np.max(state["w"])),
        "v_max":     float(np.sqrt(state["u"]**2 + state["w"]**2).max()),
        "elapsed_s": elapsed,
        "scheme":    scheme,
        "dx":        dx,
        "dt":        dt,
    }
    return grid, snapshots, elapsed, diag


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _smooth_display(field, passes=4):
    """Display-only Shapiro smoothing — does not affect simulation state."""
    f = field.copy()
    for _ in range(passes):
        f   = 0.25 * np.roll(f, -1, axis=1) + 0.5 * f + 0.25 * np.roll(f, 1, axis=1)
        pad = np.concatenate([f[[0], :], f, f[[-1], :]], axis=0)
        f   = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    return f


def plot_snapshots(grid, snapshots, diag, scheme, dx, shapiro, t_end,
                   display_smooth=4):
    snaps = sorted(snapshots, key=lambda x: x[0])
    if len(snaps) > 6:
        idx   = np.round(np.linspace(0, len(snaps) - 1, 6)).astype(int)
        snaps = [snaps[i] for i in idx]

    n_col = len(snaps)
    fig, axes = plt.subplots(2, n_col, figsize=(3.5 * n_col, 7.0))
    if n_col == 1:
        axes = axes[:, np.newaxis]

    theta_abs_max = max(np.max(np.abs(s["theta"])) for _, s in snaps)
    v_abs_max     = max(np.sqrt(s["u"]**2 + s["w"]**2).max() for _, s in snaps)
    theta_abs_max = max(theta_abs_max, 0.01)
    v_abs_max     = max(v_abs_max, 0.01)

    X = grid.x_2d / 1000.0
    Z = grid.z_2d / 1000.0

    for j, (t_s, state) in enumerate(snaps):
        theta = _smooth_display(state["theta"], passes=display_smooth) if display_smooth else state["theta"]
        vmag  = np.sqrt(state["u"]**2 + state["w"]**2)
        if display_smooth:
            vmag = _smooth_display(vmag, passes=display_smooth)

        # Row 0: theta'
        ax0 = axes[0, j]
        cf0 = ax0.contourf(X, Z, theta, levels=50, cmap="RdBu_r",
                           norm=Normalize(-theta_abs_max, theta_abs_max))
        theta_levels = np.linspace(-theta_abs_max, theta_abs_max, 11)
        ax0.contour(X, Z, theta, levels=theta_levels,
                    colors="k", linewidths=0.5, alpha=0.4)
        plt.colorbar(cf0, ax=ax0, label="theta' (K)", fraction=0.046)
        ax0.set_title("t = {:.0f} s".format(t_s), fontsize=10)
        ax0.set_xlabel("x  (km)")
        if j == 0:
            ax0.set_ylabel("z  (km)  |  theta'", fontsize=9)
        ax0.set_aspect("equal")

        # Row 1: |v|
        ax1 = axes[1, j]
        cf1 = ax1.contourf(X, Z, vmag, levels=50, cmap="YlOrRd",
                           vmin=0, vmax=v_abs_max)
        v_levels = np.linspace(0, v_abs_max, 6)
        ax1.contour(X, Z, vmag, levels=v_levels,
                    colors="k", linewidths=0.5, alpha=0.4)
        plt.colorbar(cf1, ax=ax1, label="|v| (m/s)", fraction=0.046)
        ax1.set_xlabel("x  (km)")
        if j == 0:
            ax1.set_ylabel("z  (km)  |  |v|", fontsize=9)
        ax1.set_aspect("equal")

    filter_tag = "Shapiro filter" if shapiro else "no filter"
    smooth_tag = "  ({}x display smoothing)".format(display_smooth) if display_smooth else ""
    fig.suptitle(
        "P&C (2022) Experiment 1 -- Convective Bubble\n"
        "{},  dx=dz={:.0f} m,  dt={} s,  {}{}".format(
            scheme, dx, diag["dt"], filter_tag, smooth_tag),
        fontsize=12,
    )
    plt.tight_layout()

    tag   = "{}_dx{}m{}".format(scheme, int(dx), "_shapiro" if shapiro else "")
    fname = os.path.join(OUT_DIR, "pc_exp1_benchmark_{}.png".format(tag))
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Figure saved -> {}".format(fname))
    return fname


def plot_velocity_time(grid, snapshots, diag, scheme, dx):
    times  = [t for t, _ in snapshots]
    v_maxs = [np.sqrt(s["u"]**2 + s["w"]**2).max() for _, s in snapshots]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.array(times) / 60.0, v_maxs, "b-o", markersize=4,
            label="|v|max  ({})".format(scheme))
    ax.axvline(T_BUOY * 3 / 60.0, color="gray", ls="--", lw=1,
               label="3T = {:.0f}s ({:.1f} min, paper velocity max)".format(
                   3*T_BUOY, 3*T_BUOY/60.0))
    ax.set_xlabel("Time  (min)")
    ax.set_ylabel("|v|max  (m/s)")
    ax.set_title("P&C (2022) Exp 1 -- Maximum wind speed vs time\n"
                 "{}, dx={:.0f} m".format(scheme, dx))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fname = os.path.join(OUT_DIR,
                         "pc_exp1_vmax_{}_dx{}m.png".format(scheme, int(dx)))
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print("  Velocity-time figure -> {}".format(fname))
    return fname


def print_diagnostics(diag):
    scheme = diag["scheme"]
    dx     = diag["dx"]
    print("")
    print("="*65)
    print("  P&C (2022) Experiment 1 -- Diagnostics at t={:.0f} s".format(diag["t"]))
    print("  Scheme: {}   dx={:.0f} m   dt={} s".format(scheme, dx, diag["dt"]))
    print("="*65)
    print("  theta'max = {:.4f} K".format(diag["theta_max"]))
    print("  theta'min = {:.4f} K".format(diag["theta_min"]))
    print("  wmax      = {:.4f} m/s   (paper W_scale = {:.2f} m/s)".format(
        diag["w_max"], W_SCALE))
    print("  |v|max    = {:.4f} m/s".format(diag["v_max"]))
    print("  Wall time : {:.1f} s".format(diag["elapsed_s"]))
    print("="*65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P&C (2022) Experiment 1: Convective Bubble benchmark"
    )
    parser.add_argument("--scheme", default="RK4",
                        choices=["EPI3", "EPI2", "EPI3FJ", "EPI2FJ", "RK4", "SI"],
                        help="Time integration scheme")
    parser.add_argument("--dx",    type=float, default=40.0,
                        help="Grid spacing in m (default: 40; paper uses 20)")
    parser.add_argument("--dt",    type=float, default=None,
                        help="Time step in s (default: auto)")
    parser.add_argument("--t_end", type=float, default=1200.0,
                        help="End time in s (default: 1200)")
    parser.add_argument("--diffusion", type=float, default=0.0,
                        help="Explicit diffusion coeff kappa in m^2/s (default: 0). "
                             "Use 1-3 m^2/s for RK4 runs >1000s to prevent blow-up.")
    parser.add_argument("--no-shapiro", dest="shapiro", action="store_false",
                        help="Disable Shapiro filter")
    parser.add_argument("--no-vtime",   dest="vtime",   action="store_false",
                        help="Skip velocity-vs-time plot")
    parser.set_defaults(shapiro=True, vtime=True)
    args = parser.parse_args()

    print("\n  Physical scales:")
    print("    B0 = {:.5f} m/s^2  |  T = {:.0f} s  |  W = {:.2f} m/s".format(
        _B0, T_BUOY, W_SCALE))
    print("    Velocity max expected at t ~ 3T = {:.0f} s ({:.1f} min)".format(
        3*T_BUOY, 3*T_BUOY/60.0))

    grid, snapshots, elapsed, diag = run(
        scheme  = args.scheme,
        dx      = args.dx,
        dt      = args.dt,
        t_end   = args.t_end,
        shapiro = args.shapiro,
        kappa   = args.diffusion,
    )

    print_diagnostics(diag)

    fname1 = plot_snapshots(grid, snapshots, diag,
                            args.scheme, args.dx, args.shapiro,
                            diag["t"])
    fname2 = None
    if args.vtime:
        fname2 = plot_velocity_time(grid, snapshots, diag, args.scheme, args.dx)

    print("\n  Done in {:.1f} s".format(elapsed))
    print("  Snapshots plot : {}".format(fname1))
    if fname2:
        print("  Velocity plot  : {}".format(fname2))
