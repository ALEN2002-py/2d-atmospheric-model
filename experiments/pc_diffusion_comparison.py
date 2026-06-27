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
T_END    = 900.0     # s   (> 3 * T_buoy ~ 660 s, mushroom fully formed)
SNAP_T   = [0, 100, 200, 300, 450, 600, 750, 900]   # s

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
KAPPA2_REF = 0.5      # m^2/s
KAPPA4_REF = 100.0    # m^4/s
KAPPA8_REF = 1.0e6    # m^8/s

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
def run_variant(label, dx, dt, grid_params, snap_times=SNAP_T):
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx, **grid_params}
    grid  = Grid(params)
    state = make_ic(grid)

    use_shapiro   = grid_params.pop("_shapiro", False)
    shapiro_every = max(1, int(round(30.0 / dt)))   # every ~30 s physical time

    snaps       = {}
    nstep_total = int(round(T_END / dt))
    dt_exact    = T_END / nstep_total

    # Pre-compute the step index nearest to each snap time (avoids float boundary issues)
    target_steps = {}
    for ts in snap_times:
        if ts == 0:
            snaps[0] = state["theta"].copy()
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
            snaps[target_steps[n + 1]] = state["theta"].copy()
        if not np.all(np.isfinite(state["w"])):
            print(f"  BLOW-UP at t={t_sim:.1f}s — aborting variant.", flush=True)
            break

    elapsed = wall_time.perf_counter() - t0
    th = state["theta"]
    w  = state["w"]
    stats = {
        "theta_max": float(np.max(th)),
        "theta_min": float(np.min(th)),
        "w_max":     float(np.max(w)),
        "w_min":     float(np.min(w)),
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
            th  = snaps.get(ts)

            ax.set_facecolor("#0a1628")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
                spine.set_linewidth(0.5)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="P&C Exp 1 diffusion comparison — evolution grid")
    parser.add_argument("--dx", type=float, default=20.0,
                        help="Grid spacing [m] (default 20, paper uses 20)")
    args = parser.parse_args()

    dx      = args.dx
    dt      = CFL * dx / CS
    dt_tiny = dt / 5.0

    k2 = _kappa(2, dx, dt)
    k4 = _kappa(4, dx, dt)
    k8 = _kappa(8, dx, dt)

    print(f"\n{'='*65}")
    print(f"  P&C Exp 1 diffusion comparison")
    print(f"  dx={dx} m   dt={dt:.4f} s   dt_tiny={dt_tiny:.5f} s")
    print(f"  T_buoy={T_BUOY:.0f}s   3T={3*T_BUOY:.0f}s   W_scale={W_SCAL:.2f} m/s")
    print(f"  kappa2={k2:.2f}  kappa4={k4:.1f}  kappa8={k8:.2e}")
    print(f"{'='*65}\n")

    variants = [
        ("IDEAL  (no diffusion)",
         dt,  {}),
        (f"nabla2  (k2={k2:.2g} m2/s)",
         dt,  {"diffusion_coeff": k2, "diffusion_order": 2}),
        (f"nabla4  (k4={k4:.2g} m4/s)",
         dt,  {"diffusion_coeff": k4, "diffusion_order": 4}),
        (f"nabla8  (k8={k8:.2e} m8/s)",
         dt,  {"diffusion_coeff": k8, "diffusion_order": 8}),
        ("Shapiro (every ~30 s)",
         dt,  {"_shapiro": True}),
        ("IDEAL tiny dt (ref)",
         dt_tiny, {}),
    ]

    all_snaps = []
    all_stats = []
    labels    = []
    for label, vdt, gp in variants:
        print(f"Running: {label}", flush=True)
        snaps, stats = run_variant(label, dx, vdt, dict(gp))
        stats["dt"] = vdt
        all_snaps.append(snaps)
        all_stats.append(stats)
        labels.append(label)

    out = os.path.join(OUT_DIR, f"pc_diffcomp_evolution_dx{int(dx)}m.png")
    plot_evolution_grid(all_snaps, labels, dx, out)
    print_summary_table(labels, all_stats, dx)


if __name__ == "__main__":
    main()
