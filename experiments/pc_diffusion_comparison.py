"""
pc_diffusion_comparison.py
==========================
P&C (2022) Experiment 1: Convective Bubble — diffusion strategy comparison.

Mirrors diffusion_comparison.py (G&R Case 2) but for the 5km x 5km P&C domain.

Produces a 6-row x 8-column evolution figure:

  Rows  (one per variant):
    1. IDEAL         — RK4, standard CFL dt, NO diffusion
    2. nabla2        — RK4 + Laplacian       kappa2 * nabla^2
    3. nabla4        — RK4 + biharmonic     -kappa4 * nabla^4
    4. nabla8        — RK4 + octaharmonic   -kappa8 * nabla^8
    5. Shapiro       — RK4 + Shapiro (1-2-1) filter every ~30 s
    6. IDEAL (tiny)  — RK4, dt/5 (5x more accurate reference)

  Columns (one per snapshot time):
    t = 0, 100, 200, 300, 450, 600, 750, 900 s

P&C Exp 1 parameters (Section 6.1, Pudykiewicz & Clancy 2022):
  Domain    : 5000 m x 5000 m
  Base state: isentropic, theta_bar = 300 K
  Bubble IC : theta' = AT                         for r <= a
              theta' = AT * exp(-(r-a)^2/(2s^2))  for r > a
              AT = 0.5 K,  a = 400 m,  sigma = 100 m
              Centre: (2500 m, 700 m)
  Physical scales: T_buoy ~ 220 s,  W_scale ~ 3.6 m/s
  Velocity max at 3T ~ 660 s

USAGE
-----
  python experiments/pc_diffusion_comparison.py           # dx=20m (recommended)
  python experiments/pc_diffusion_comparison.py --dx 40   # dx=40m (fast, ~15 min)
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
# P&C Exp 1 parameters (Section 6.1)
# ---------------------------------------------------------------------------
LX, LZ  = 5000.0, 5000.0
AT       = 0.5       # K   bubble amplitude
A_RAD    = 400.0     # m   flat-top radius (= 20 * delta, delta=20m)
SIGMA    = 100.0     # m   Gaussian edge width (= 5 * delta)
X0       = 2500.0    # m   bubble centre x
Z0       = 700.0     # m   bubble centre z
T_END    = 1200.0    # s   IDEAL blows up at ~1126s; others clean at 1200s (cap at z~3km)
SNAP_T   = [0, 300, 600, 900, 1100, 1200]            # s  (6 snapshots)

# Physical scales (for reference)
_G     = 9.81
_TBAR  = 300.0
_B0    = _G * AT / _TBAR          # ~ 0.01635 m/s^2
_L     = 2.0 * A_RAD              # = 800 m
T_BUOY = np.sqrt(_L / _B0)        # ~ 220 s
W_SCAL = np.sqrt(_B0 * _L)        # ~ 3.6 m/s

# Contour levels for theta' — same range as bubble amplitude
CLEV = np.arange(0.02, 0.501, 0.02)

# CFL settings
CS  = 347.0   # m/s  (sqrt(gamma*Rd*T0), T0=300K)
CFL = 0.34    # max acoustic CFL for RK4

# ---------------------------------------------------------------------------
# kappa reference values at dx=10m — same as G&R comparison
# (will be scaled to actual dx inside _kappa)
# ---------------------------------------------------------------------------
# P&C bubble scale (a=400m) is larger than G&R (r_c=250m),
# so use gentler diffusion — half the G&R reference values.
KAPPA2_REF = 0.167    # m^2/s  (reduced 3x from 0.5)
KAPPA4_REF = 16.7     # m^4/s  (reduced 6x from 100.0 — nabla4 over-damped at dx=20m)
KAPPA8_REF = 3.33e5   # m^8/s  (reduced 3x from 1.0e6)

def _kappa(order, dx, dt):
    """Scale kappa with dx, capped at RK4 explicit stability limit.
    Max eigenvalue of discrete 2D Laplacian = 8/dx^2.
    For (nabla^2)^n: eigenvalue ~ (8/dx^2)^n.
    RK4 stability: kappa*(8/dx^2)^n*dt <= 2.79. Safety factor 70%.
    """
    base = {2: KAPPA2_REF, 4: KAPPA4_REF, 8: KAPPA8_REF}[order]
    raw  = base * (dx / 10.0) ** order
    n         = order // 2
    kappa_max = 0.7 * 2.79 * dx**order / (8**n * dt)
    return min(raw, kappa_max)

# ---------------------------------------------------------------------------
# Initial condition — cylindrical flat-top + Gaussian edge (P&C eq. 6.1)
# ---------------------------------------------------------------------------
def make_ic(grid):
    """P&C (2022) Experiment 1 initial condition: flat-top bubble with Gaussian edge.

    theta' = AT                            for r <= A_RAD   (flat cylindrical core)
    theta' = AT * exp(-(r-A_RAD)^2/2s^2)  for r >  A_RAD   (smooth Gaussian tail)
    """
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X0)**2 + (grid.z_2d - Z0)**2)
    state["theta"] = np.where(
        r <= A_RAD,
        AT,
        AT * np.exp(-(r - A_RAD)**2 / (2.0 * SIGMA**2)),
    )
    return state

# ---------------------------------------------------------------------------
# Run one variant, saving snapshots at SNAP_T
# ---------------------------------------------------------------------------
def run_variant(label, dx, dt, grid_params, snap_times=SNAP_T, shapiro_period=30.0):
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx, **grid_params}
    grid  = Grid(params)
    state = make_ic(grid)

    use_shapiro   = grid_params.pop("_shapiro", False)
    if not shapiro_period or shapiro_period <= dt:
        shapiro_every = 1
    else:
        shapiro_every = max(1, int(round(shapiro_period / dt)))

    snaps       = {}
    nstep_total = int(round(T_END / dt))
    dt_exact    = T_END / nstep_total

    # Pre-compute the step index nearest to each snap time (avoids float boundary issues)
    target_steps = {}
    for ts in snap_times:
        if ts == 0:
            snaps[0] = {"theta": state["theta"].copy(),
                        "u": state["u"].copy(), "w": state["w"].copy()}
        else:
            idx = int(round(ts / dt_exact))
            idx = min(idx, nstep_total)   # clamp to last step
            target_steps[idx] = ts

    t0 = wall_time.perf_counter()
    for n in range(nstep_total):
        state_new, state, _ = step(state, grid, dt_exact, scheme="RK4")
        state = state_new
        if use_shapiro and (n + 1) % shapiro_every == 0:
            state = shapiro_filter(state, grid)
        t_sim = (n + 1) * dt_exact
        if (n + 1) in target_steps:
            snaps[target_steps[n + 1]] = {
                    "theta": state["theta"].copy(),
                    "u":     state["u"].copy(),
                    "w":     state["w"].copy(),
                }
        if not np.all(np.isfinite(state["w"])):
            print(f"  BLOW-UP at t={t_sim:.1f}s — aborting variant.", flush=True)
            break

    elapsed = wall_time.perf_counter() - t0

    # Compute stats from last VALID snapshot (avoids nan on blow-up)
    if snaps:
        last_valid = snaps[max(snaps.keys())]
        th_stat = last_valid["theta"]
        w_stat  = last_valid["w"]
    else:
        th_stat = state["theta"]
        w_stat  = state["w"]

    stats = {
        "theta_max": float(np.nanmax(th_stat)),
        "theta_min": float(np.nanmin(th_stat)),
        "w_max":     float(np.nanmax(w_stat)),
        "w_min":     float(np.nanmin(w_stat)),
        "wall":      elapsed,
        "dt":        dt,
        "nstep":     nstep_total,
    }
    print(f"  {label:<38} theta_max={stats['theta_max']:.3f} K  "
          f"w_max={stats['w_max']:.3f} m/s  wall={elapsed:.1f}s", flush=True)
    return snaps, stats

# ---------------------------------------------------------------------------
# Evolution grid figure
# ---------------------------------------------------------------------------
def plot_evolution_grid(all_snaps, variant_labels, dx, out_path):
    try:
        from scipy.ndimage import gaussian_filter
        SMOOTH = True
    except ImportError:
        SMOOTH = False

    n_rows = len(all_snaps)
    n_cols = len(SNAP_T)

    PAD_L  = 1.8
    CELL_W = 1.55
    CELL_H = 1.55
    PAD_R  = 0.70
    PAD_T  = 0.65
    PAD_B  = 0.30

    fig_w = PAD_L + n_cols * CELL_W + PAD_R
    fig_h = PAD_T + n_rows * CELL_H + PAD_B

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
    )

    plt.subplots_adjust(
        left   = PAD_L / fig_w,
        right  = (PAD_L + n_cols * CELL_W) / fig_w,
        top    = 1.0 - PAD_T / fig_h,
        bottom = PAD_B / fig_h,
        hspace = 0.04,
        wspace = 0.04,
    )

    cmap  = "RdYlBu_r"
    vmin, vmax = CLEV[0], CLEV[-1]
    extent = [0, LX / 1000, 0, LZ / 1000]   # km

    for row, (snaps, label) in enumerate(zip(all_snaps, variant_labels)):
        for col, ts in enumerate(SNAP_T):
            ax  = axes[row][col]

            ax.set_facecolor("#0a1628")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
                spine.set_linewidth(0.5)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

            snap = snaps.get(ts)
            th   = snap["theta"] if snap is not None else None
            u_f  = snap["u"]     if snap is not None else None
            w_f  = snap["w"]     if snap is not None else None

            if th is not None:
                disp = gaussian_filter(th, sigma=0.9) if SMOOTH else th
                ax.imshow(
                    disp, origin="lower", extent=extent,
                    aspect="auto", cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    interpolation="bicubic",
                )
                x_km = np.linspace(0, LX / 1000, th.shape[1])
                z_km = np.linspace(0, LZ / 1000, th.shape[0])
                X, Z = np.meshgrid(x_km, z_km)
                ax.contour(X, Z, disp, levels=CLEV,
                           colors="white", linewidths=0.35, alpha=0.55)

                # Velocity quiver — subsample every ~160 m (8 cells at dx=20m)
                skip = max(1, int(round(320.0 / dx)))
                ax.quiver(
                    X[::skip, ::skip], Z[::skip, ::skip],
                    u_f[::skip, ::skip], w_f[::skip, ::skip],
                    color="white", alpha=0.55,
                    scale=40, width=0.004,
                    headwidth=3, headlength=4,
                )

            ax.set_xlim(0, LX / 1000)
            ax.set_ylim(0, LZ / 1000)

            if row == 0:
                ax.set_title(f"t = {ts} s", fontsize=8.5,
                             fontweight="bold", color="#222", pad=4)

            if row == n_rows - 1:
                ax.tick_params(labelbottom=True)
                ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
                ax.tick_params(axis="x", labelsize=6.5)
                if col == 0:
                    ax.set_xlabel("x [km]", fontsize=7, labelpad=2)

            if col == 0:
                ax.tick_params(labelleft=True)
                ax.set_ylabel("z [km]", fontsize=7, labelpad=2)
                ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
                ax.tick_params(axis="y", labelsize=6.5)
                ax.text(
                    -0.38, 0.5, label,
                    transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", color="#111",
                    ha="right", va="center", rotation=0,
                )

    # Colorbar
    cbar_left   = (PAD_L + n_cols * CELL_W + 0.12) / fig_w
    cbar_bottom = PAD_B / fig_h + 0.04
    cbar_width  = 0.20 / fig_w
    cbar_height = 1.0 - (PAD_T + PAD_B) / fig_h - 0.06

    sm = plt.cm.ScalarMappable(cmap=cmap,
         norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\theta'$ [K]", fontsize=10, labelpad=6)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        f"P&C Exp 1 — Convective Bubble   (dx = {dx} m,  t_end = {T_END:.0f} s)",
        fontsize=11, fontweight="bold", color="#111",
        y=1.0 - 0.08 / fig_h,
    )

    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  Saved: {out_path}", flush=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(labels, all_stats, dx):
    W = 95
    ref_th = all_stats[-1]["theta_max"]
    ref_w  = all_stats[-1]["w_max"]
    c_var = "Variant"; c_dt = "dt (s)"; c_thmax = "th'max(K)"
    c_thmin = "th'min(K)"; c_wmax = "wmax(m/s)"; c_wall = "Wall(s)"
    sep  = "+" + "-"*42 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*9 + "+"
    hdr  = (f"| {c_var:<40} | {c_dt:>8} | {c_thmax:>8} "
            f"| {c_thmin:>8} | {c_wmax:>8} | {c_wall:>7} |")

    print("\n" + "="*W)
    print(f"  SUMMARY TABLE  --  P&C Exp 1, dx={dx} m, t={T_END:.0f} s")
    print(f"  T_buoy={T_BUOY:.0f}s  W_scale={W_SCAL:.2f} m/s  3T={3*T_BUOY:.0f}s")
    print(f"  Reference (IDEAL tiny dt):  th'_max = {ref_th:.4f} K,  w_max = {ref_w:.4f} m/s")
    print("="*W)
    print(sep); print(hdr); print(sep)
    for label, s in zip(labels, all_stats):
        print(f"| {label:<40} | {s['dt']:>8.4f} | {s['theta_max']:>8.4f} "
              f"| {s['theta_min']:>8.4f} | {s['w_max']:>8.4f} | {s['wall']:>7.1f} |")
    print(sep); print()

    c_dth = "Dth'max(K)"; c_dw = "Dwmax(m/s)"
    sep2 = "+" + "-"*42 + "+" + "-"*12 + "+" + "-"*12 + "+"
    hdr2 = f"| {c_var:<40} | {c_dth:>10} | {c_dw:>10} |"
    print("  Deviation from IDEAL tiny-dt reference:")
    print(sep2); print(hdr2); print(sep2)
    for label, s in zip(labels, all_stats):
        dth = s["theta_max"] - ref_th
        dw  = s["w_max"]     - ref_w
        print(f"| {label:<40} | {dth:>+10.4f} | {dw:>+10.4f} |")
    print(sep2); print()

# ---------------------------------------------------------------------------
# Vertical profile plot — theta' at x = X0 m, t = T_END s
# ---------------------------------------------------------------------------
def plot_vertical_profile(all_snaps, variant_labels, dx, out_path):
    STYLES = [
        {"color": "#1a1a1a", "lw": 2.2, "ls": "-",           "zorder": 10},  # IDEAL
        {"color": "#d62728", "lw": 1.6, "ls": "--",          "zorder":  6},  # nabla2
        {"color": "#ff7f0e", "lw": 1.6, "ls": "-.",          "zorder":  7},  # nabla4
        {"color": "#2ca02c", "lw": 1.6, "ls": ":",           "zorder":  8},  # nabla8
        {"color": "#9467bd", "lw": 1.6, "ls": (0, (4, 1.5)), "zorder":  9},  # Shapiro
        {"color": "#17becf", "lw": 1.8, "ls": (0, (6, 2)),   "zorder": 11},  # IDEAL tiny
    ]

    nz = int(round(LZ / dx))
    z_km = (np.arange(nz) + 0.5) * dx / 1000.0

    nx = int(round(LX / dx))
    x_centres = (np.arange(nx) + 0.5) * dx
    ix = int(np.argmin(np.abs(x_centres - X0)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for i, (snaps, label, style) in enumerate(zip(all_snaps, variant_labels, STYLES)):
        snap = snaps.get(T_END)
        if snap is None:
            # Variant blew up before T_END — skip rather than show stale data
            ax.plot([], [], label=f"{label}  [unstable]",
                    color=style["color"], lw=style["lw"],
                    ls=":", zorder=style["zorder"])
            continue
        th = snap["theta"] if isinstance(snap, dict) else snap
        profile = th[:, ix]
        ax.plot(profile, z_km, label=label,
                color=style["color"], lw=style["lw"],
                ls=style["ls"], zorder=style["zorder"])

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z [km]", fontsize=12)
    ax.set_title(
        f"Vertical profile of $\\theta'$ at $x = {X0:.0f}$ m,  $t = {int(T_END)}$ s\n"
        f"P&C Exp 1  (dx = {dx} m)  —  IDEAL unstable after t ≈ 1126 s",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 0.70)
    ax.set_ylim(0.0, LZ / 1000.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color="#ddd", lw=0.6, zorder=0)
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9,
              edgecolor="#ccc", handlelength=2.8)

    fig.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Final snapshot — all variants at t = T_END  (2 × 3 grid)
# ---------------------------------------------------------------------------
def plot_final_snapshot(all_snaps, variant_labels, dx, out_path):
    try:
        from scipy.ndimage import gaussian_filter
        SMOOTH = True
    except ImportError:
        SMOOTH = False

    n_vars = len(all_snaps)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    CELL  = 3.6
    PAD_L = 0.55; PAD_R = 0.80; PAD_T = 0.55; PAD_B = 0.45

    fig_w = PAD_L + n_cols * CELL + PAD_R
    fig_h = PAD_T + n_rows * CELL + PAD_B

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True)
    plt.subplots_adjust(
        left=PAD_L/fig_w, right=(PAD_L+n_cols*CELL)/fig_w,
        top=1.0-PAD_T/fig_h, bottom=PAD_B/fig_h,
        hspace=0.08, wspace=0.06,
    )

    cmap       = "RdYlBu_r"
    vmin, vmax = CLEV[0], CLEV[-1]
    extent     = [0, LX/1000, 0, LZ/1000]

    for idx, (snaps, label) in enumerate(zip(all_snaps, variant_labels)):
        row = idx // n_cols
        col = idx  % n_cols
        ax  = axes[row][col]
        ax.set_facecolor("#0a1628")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334"); spine.set_linewidth(0.6)

        snap = snaps.get(T_END)
        blew_up = (snap is None)   # variant blew up before T_END
        if blew_up:
            snap = snaps[max(snaps.keys())]

        th  = snap["theta"] if isinstance(snap, dict) else snap
        u_f = snap["u"]     if isinstance(snap, dict) else None
        w_f = snap["w"]     if isinstance(snap, dict) else None

        if blew_up:
            # Show solid black panel with a BLOW-UP label — do NOT show stale data
            ax.set_facecolor("#000000")
            ax.text(0.5, 0.5, "NUMERICAL\nINSTABILITY",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="#ff4444",
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#1a1a1a",
                              ec="#ff4444", lw=1.5))
        else:
            disp = gaussian_filter(th, sigma=0.9) if SMOOTH else th
            ax.imshow(disp, origin="lower", extent=extent, aspect="auto",
                      cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic")

            x_km = np.linspace(0, LX/1000, th.shape[1])
            z_km = np.linspace(0, LZ/1000, th.shape[0])
            X, Z = np.meshgrid(x_km, z_km)
            ax.contour(X, Z, disp, levels=CLEV, colors="white",
                       linewidths=0.4, alpha=0.55)

            if u_f is not None:
                skip = max(1, int(round(320.0 / dx)))
                ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                          u_f[::skip, ::skip], w_f[::skip, ::skip],
                          color="white", alpha=0.6, scale=40, width=0.004,
                          headwidth=3, headlength=4)

        ax.set_xlim(0, LX/1000); ax.set_ylim(0, LZ/1000)
        if row == n_rows - 1:
            ax.tick_params(labelbottom=True)
            ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("x [km]", fontsize=9, labelpad=3)
        if col == 0:
            ax.tick_params(labelleft=True)
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
            ax.tick_params(axis="y", labelsize=8)
            ax.set_ylabel("z [km]", fontsize=9, labelpad=3)
        ax.set_title(label, fontsize=9.5, fontweight="bold",
                     color="#111", pad=5)

    cbar_left   = (PAD_L + n_cols*CELL + 0.12) / fig_w
    cbar_bottom = PAD_B/fig_h + 0.04
    cbar_width  = 0.22/fig_w
    cbar_height = 1.0 - (PAD_T+PAD_B)/fig_h - 0.06
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\theta'$ [K]", fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        f"P&C Exp 1 — Convective Bubble at t = {int(T_END)} s   (dx = {dx} m)",
        fontsize=12, fontweight="bold", color="#111", y=1.0-0.10/fig_h,
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be at module level for pickling on Windows)
# ---------------------------------------------------------------------------
def _worker(args):
    """Unpack and run one variant — used by multiprocessing.Pool."""
    label, dx, vdt, gp, snap_times, sp = args
    snaps, stats = run_variant(label, dx, vdt, dict(gp),
                               snap_times=snap_times, shapiro_period=sp)
    stats["dt"] = vdt
    return label, snaps, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P&C Exp 1 diffusion comparison — evolution grid")
    parser.add_argument("--dx", type=float, default=20.0,
                        help="Grid spacing [m] (default 20, paper uses 20)")
    parser.add_argument("--shapiro-period", type=float, default=30.0,
                        help="Shapiro filter interval in simulation seconds "
                             "(default 30; use 0 for every step)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run variants in parallel using multiprocessing")
    parser.add_argument("--no-ref", action="store_true",
                        help="Skip IDEAL tiny-dt reference (saves ~50%% of run time)")
    args = parser.parse_args()

    dx      = args.dx
    dt      = CFL * dx / CS
    dt_tiny = dt / 5.0
    sp = args.shapiro_period if args.shapiro_period > 0 else None

    k2 = _kappa(2, dx, dt)
    k4 = _kappa(4, dx, dt)
    k8 = _kappa(8, dx, dt)

    print(f"\n{'='*65}")
    print(f"  P&C Exp 1 diffusion comparison")
    print(f"  dx={dx} m   dt={dt:.4f} s   dt_tiny={dt_tiny:.5f} s")
    print(f"  T_buoy={T_BUOY:.0f}s   3T={3*T_BUOY:.0f}s   W_scale={W_SCAL:.2f} m/s")
    print(f"  kappa2={k2:.2f}  kappa4={k4:.1f}  kappa8={k8:.2e}")
    print(f"  parallel={'yes' if args.parallel else 'no'}")
    print(f"  no-ref={'yes' if args.no_ref else 'no'}")
    print(f"{'='*65}\n")

    variants = [
        ("IDEAL  (no diffusion)",          dt,      {}),
        (f"nabla2  (k2={k2:.2g} m2/s)",   dt,      {"diffusion_coeff": k2, "diffusion_order": 2}),
        (f"nabla4  (k4={k4:.2g} m4/s)",   dt,      {"diffusion_coeff": k4, "diffusion_order": 4}),
        (f"nabla8  (k8={k8:.2e} m8/s)",   dt,      {"diffusion_coeff": k8, "diffusion_order": 8}),
        (f"Shapiro (every {sp if sp else 'step'} s)", dt, {"_shapiro": True}),
    ]
    if not args.no_ref:
        variants.append(("IDEAL tiny dt (ref)", dt_tiny, {}))

    worker_args = [(label, dx, vdt, dict(gp), SNAP_T, sp)
                   for label, vdt, gp in variants]

    if args.parallel:
        import multiprocessing
        n_cores = min(len(variants), multiprocessing.cpu_count())
        print(f"  Spawning {n_cores} workers ...\n", flush=True)
        with multiprocessing.Pool(processes=n_cores) as pool:
            results = pool.map(_worker, worker_args)
    else:
        results = [_worker(a) for a in worker_args]

    labels, all_snaps, all_stats = zip(*results)
    labels    = list(labels)
    all_snaps = list(all_snaps)
    all_stats = list(all_stats)

    # ------------------------------------------------------------------
    # Print stats table
    # ------------------------------------------------------------------
    W = 100
    c_var = "Variant"; c_dt = "dt [s]"; c_thmax = "th'max[K]"; c_thmin = "th'min[K]"
    c_wmax = "wmax[m/s]"; c_wall = "wall[s]"
    sep = "+" + "-"*42 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*9 + "+"
    hdr = f"| {c_var:<40} | {c_dt:>8} | {c_thmax:>8} | {c_thmin:>8} | {c_wmax:>8} | {c_wall:>7} |"

    # Use IDEAL tiny-dt as reference if present; else use IDEAL (no diffusion)
    # If IDEAL blew up (theta_max=nan), fall back to Shapiro as reference
    ref_label  = "IDEAL tiny dt (ref)" if not args.no_ref else "IDEAL  (no diffusion)"
    ref_idx    = next((i for i, l in enumerate(labels) if l == ref_label), 0)
    ref_th  = all_stats[ref_idx]["theta_max"]
    ref_w   = all_stats[ref_idx]["w_max"]
    if not np.isfinite(ref_th):
        # IDEAL blew up — use nabla8 (least-diffusive stable variant) as fallback reference
        fb_label = next((l for l in labels if "nabla8" in l), labels[-1])
        fb_idx   = labels.index(fb_label)
        ref_th   = all_stats[fb_idx]["theta_max"]
        ref_w    = all_stats[fb_idx]["w_max"]
        ref_label = f"{fb_label}  [IDEAL unstable — used as fallback ref]"

    print(f"\n  Reference ({ref_label}):  th'_max = {ref_th:.4f} K,  w_max = {ref_w:.4f} m/s")
    print("="*W)
    print(sep); print(hdr); print(sep)
    for label, s in zip(labels, all_stats):
        print(f"| {label:<40} | {s['dt']:>8.4f} | {s['theta_max']:>8.4f} "
              f"| {s['theta_min']:>8.4f} | {s['w_max']:>8.4f} | {s['wall']:>7.1f} |")
    print(sep); print()

    c_dth = "Dth'max(K)"; c_dw = "Dwmax(m/s)"
    sep2 = "+" + "-"*42 + "+" + "-"*12 + "+" + "-"*12 + "+"
    hdr2 = f"| {c_var:<40} | {c_dth:>10} | {c_dw:>10} |"
    print("  Deviation from reference:")
    print(sep2); print(hdr2); print(sep2)
    for label, s in zip(labels, all_stats):
        dth = s["theta_max"] - ref_th
        dw  = s["w_max"]     - ref_w
        print(f"| {label:<40} | {dth:>+10.4f} | {dw:>+10.4f} |")
    print(sep2); print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    suffix = f"dx{int(dx)}m"

    # Evolution grid
    evo_path = os.path.join(OUT_DIR, f"pc_diffusion_evo_{suffix}.png")
    plot_evolution_grid(all_snaps, labels, dx, evo_path)

    # Final snapshot panel
    snap_path = os.path.join(OUT_DIR, f"pc_diffusion_snap_{suffix}.png")
    plot_final_snapshot(all_snaps, labels, dx, snap_path)

    # Vertical profile
    vprof_path = os.path.join(OUT_DIR, f"pc_diffusion_vprof_{suffix}.png")
    plot_vertical_profile(all_snaps, labels, dx, vprof_path)

    print(f"\n  All output saved to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
