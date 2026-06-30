"""
diffusion_comparison.py
=======================
G&R Case 2: Rising Thermal Bubble — diffusion strategy comparison.

Produces a 6-row x 8-column evolution figure:

  Rows  (one per variant):
    1. IDEAL         — RK4, standard CFL dt, NO diffusion
    2. nabla2        — RK4 + Laplacian       kappa2 * nabla^2
    3. nabla4        — RK4 + biharmonic     -kappa4 * nabla^4
    4. nabla8        — RK4 + octaharmonic   -kappa8 * nabla^8
    5. Shapiro       — RK4 + Shapiro (1-2-1) filter every ~30 s
    6. IDEAL (tiny)  — RK4, dt/10 (10x more accurate reference)

  Columns (one per snapshot time):
    t = 0, 100, 200, 300, 400, 500, 600, 700 s

Each panel shows the theta' contour field with G&R Fig-3 contour levels.

USAGE
-----
  python experiments/diffusion_comparison.py           # dx=10m (slow, accurate)
  python experiments/diffusion_comparison.py --dx 20   # dx=20m (recommended)
  python experiments/diffusion_comparison.py --dx 40   # dx=40m (fast, coarse)
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
# G&R Case 2 parameters (Section 3.2, paper page 3855)
# ---------------------------------------------------------------------------
LX, LZ   = 1000.0, 1000.0   # domain size [m]
THETA_C  = 0.5      # K   bubble amplitude (G&R 2008 eq 3.1)
R_C      = 250.0    # m   bubble radius (cosine bell extends to 250 m from centre)
X_C      = 500.0    # m   bubble centre x (middle of domain)
Z_C      = 350.0    # m   bubble centre z (below mid-domain to give room to rise)
T_END    = 700.0    # s   simulation end time (mushroom cap fully formed by ~700 s)
SNAP_T   = [0, 100, 200, 300, 400, 500, 600, 700]   # s  snapshot times

# Contour levels matching G&R Figure 3 (page 3866)
CLEV = np.arange(0.05, 0.526, 0.025)

# Speed of sound — sets the acoustic CFL constraint
CS  = 347.0   # m/s  (sqrt(gamma*Rd*T0), T0=300 K)
CFL = 0.34    # max acoustic CFL for RK4 stability

# ---------------------------------------------------------------------------
# kappa values at reference dx=10 m (stable for RK4 at CFL=0.34)
# ---------------------------------------------------------------------------
# Damping timescale at 2*dx wave: tau = 1 / (kappa * k_max^order)
# k_max = pi/dx.  Values chosen so tau ~ 50-100 s at dx=10 m.
KAPPA2_REF = 1.0      # m^2/s
KAPPA4_REF = 200.0    # m^4/s
KAPPA8_REF = 2.0e6    # m^8/s   (conservative: well within RK4 stability)

def _kappa(order, dx, dt):
    """Scale kappa to dx, capped at the RK4 explicit stability limit.

    The max eigenvalue of the discrete 2D Laplacian (nabla^2) at the
    Nyquist wavenumber is  4/dx^2 + 4/dz^2 = 8/dx^2  (for square grid).
    For (nabla^2)^n, the max eigenvalue magnitude is (8/dx^2)^n.
    RK4 stability:  kappa * (8/dx^2)^n * dt  <=  2.79
    We use a 70% safety factor.
    """
    base = {2: KAPPA2_REF, 4: KAPPA4_REF, 8: KAPPA8_REF}[order]
    raw  = base * (dx / 10.0) ** order
    n         = order // 2
    kappa_max = 0.7 * 2.79 * dx**order / (8**n * dt)
    return min(raw, kappa_max)

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
def make_ic(grid):
    """Cosine-bell thermal bubble (G&R eq 3.1)."""
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - X_C)**2 + (grid.z_2d - Z_C)**2)
    state["theta"] = np.where(
        r <= R_C,
        0.5 * THETA_C * (1.0 + np.cos(np.pi * r / R_C)),
        0.0
    )
    return state

# ---------------------------------------------------------------------------
# Run one variant, saving snapshots at SNAP_T
# ---------------------------------------------------------------------------
def run_variant(label, dx, dt, grid_params, snap_times=SNAP_T, shapiro_period=30.0):
    """
    Run G&R Case 2 with given grid_params, saving theta fields at snap_times.

    grid_params keys:
      diffusion_coeff  — kappa
      diffusion_order  — 2, 4, or 8
      _shapiro         — True to apply Shapiro filter

    shapiro_period: simulation seconds between filter applications.
                   None or 0 → every step.
    """
    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx, **grid_params}
    grid  = Grid(params)
    state = make_ic(grid)

    use_shapiro = grid_params.pop("_shapiro", False)
    if not shapiro_period or shapiro_period <= dt:
        shapiro_every = 1
    else:
        shapiro_every = max(1, int(round(shapiro_period / dt)))

    snaps       = {}
    nstep_total = int(round(T_END / dt))
    dt_exact    = T_END / nstep_total

    # Pre-compute nearest step index for each snap time (avoids float boundary issues)
    target_steps = {}
    for ts in snap_times:
        if ts == 0:
            snaps[0] = state["theta"].copy()
        else:
            idx = int(round(ts / dt_exact))
            idx = min(idx, nstep_total)
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
          f"w_max={stats['w_max']:.3f} m/s  wall={elapsed:.1f}s",
          flush=True)
    return snaps, stats

# ---------------------------------------------------------------------------
# Evolution grid figure
# ---------------------------------------------------------------------------
def plot_evolution_grid(all_snaps, variant_labels, dx, out_path):
    """
    6 rows x 8 columns polished evolution grid.
    Uses bicubic-interpolated imshow for smooth fills even at coarse dx.
    """
    try:
        from scipy.ndimage import gaussian_filter
        SMOOTH = True
    except ImportError:
        SMOOTH = False

    n_rows = len(all_snaps)
    n_cols = len(SNAP_T)

    PAD_L  = 1.8   # inches reserved for row labels on the left
    CELL_W = 1.55  # inches per panel
    CELL_H = 1.55
    PAD_R  = 0.70  # for colorbar
    PAD_T  = 0.65  # top margin (room for suptitle above column headers)
    PAD_B  = 0.30  # bottom margin

    fig_w = PAD_L + n_cols * CELL_W + PAD_R
    fig_h = PAD_T + n_rows * CELL_H + PAD_B

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True
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

            # Clean frame
            ax.set_facecolor("#0a1628")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
                spine.set_linewidth(0.5)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

            if th is not None:
                # Smooth for display only (never touches simulation state)
                disp = gaussian_filter(th, sigma=0.9) if SMOOTH else th

                # Bicubic-interpolated filled colour — smooth even at dx=40m
                ax.imshow(
                    disp, origin="lower", extent=extent,
                    aspect="auto", cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    interpolation="bicubic",
                )

                # Overlay crisp contour lines
                x_km = np.linspace(0, LX / 1000, th.shape[1])
                z_km = np.linspace(0, LZ / 1000, th.shape[0])
                X, Z = np.meshgrid(x_km, z_km)
                ax.contour(X, Z, disp, levels=CLEV,
                           colors="white", linewidths=0.35, alpha=0.55)

            ax.set_xlim(0, LX / 1000)
            ax.set_ylim(0, LZ / 1000)

            # --- Column header (top row only) ---
            if row == 0:
                ax.set_title(f"t = {ts} s", fontsize=8.5,
                             fontweight="bold", color="#222", pad=4)

            # --- x label (bottom row, leftmost panel only) ---
            if row == n_rows - 1:
                ax.tick_params(labelbottom=True)
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax.tick_params(axis="x", labelsize=6.5)
                if col == 0:
                    ax.set_xlabel("x [km]", fontsize=7, labelpad=2)

            # --- y label (left column only) ---
            if col == 0:
                ax.tick_params(labelleft=True)
                ax.set_ylabel("z [km]", fontsize=7, labelpad=2)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax.tick_params(axis="y", labelsize=6.5)

            # --- Row label (outside left edge of leftmost panel) ---
            if col == 0:
                ax.text(
                    -0.38, 0.5, label,
                    transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", color="#111",
                    ha="right", va="center",
                    rotation=0,
                )

    # --- Colorbar (explicit axes, far right) ---
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

    # --- Title (single line, above column headers) ---
    fig.suptitle(
        f"G&R Case 2 — Rising Thermal Bubble   (dx = {dx} m,  t_end = {T_END:.0f} s)",
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
    """Print a formatted table of final diagnostics for all variants."""
    W = 95
    sep  = "+" + "-"*42 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*9 + "+"
    c_var = "Variant";  c_dt = "dt (s)"; c_thmax = "th'max(K)"
    c_thmin = "th'min(K)"; c_wmax = "wmax(m/s)"; c_wall = "Wall(s)"
    hdr = (f"| {c_var:<40} | {c_dt:>8} | {c_thmax:>8} "
           f"| {c_thmin:>8} | {c_wmax:>8} | {c_wall:>7} |")

    ref_th = all_stats[-1]["theta_max"]
    ref_w  = all_stats[-1]["w_max"]

    print("\n" + "="*W)
    print(f"  SUMMARY TABLE  --  G&R Case 2, dx={dx} m, t=700 s")
    print(f"  Reference (IDEAL tiny dt):  th'_max = {ref_th:.4f} K,  w_max = {ref_w:.4f} m/s")
    print("="*W)
    print(sep)
    print(hdr)
    print(sep)
    for label, s in zip(labels, all_stats):
        print(f"| {label:<40} | {s['dt']:>8.4f} | {s['theta_max']:>8.4f} "
              f"| {s['theta_min']:>8.4f} | {s['w_max']:>8.4f} | {s['wall']:>7.1f} |")
    print(sep)
    print()

    # Deviation from reference
    c_dth = "Dth'max(K)"; c_dw = "Dwmax(m/s)"
    sep2 = "+" + "-"*42 + "+" + "-"*12 + "+" + "-"*12 + "+"
    hdr2 = f"| {c_var:<40} | {c_dth:>10} | {c_dw:>10} |"
    print("  Deviation from IDEAL tiny-dt reference:")
    print(sep2)
    print(hdr2)
    print(sep2)
    for label, s in zip(labels, all_stats):
        dth = s["theta_max"] - ref_th
        dw  = s["w_max"]     - ref_w
        print(f"| {label:<40} | {dth:>+10.4f} | {dw:>+10.4f} |")
    print(sep2)
    print()


# ---------------------------------------------------------------------------
# Vertical profile plot — theta' at x = 500 m, t = 700 s
# ---------------------------------------------------------------------------
def plot_vertical_profile(all_snaps, variant_labels, dx, out_path):
    """
    Overlay the vertical profile of theta' at x = 500 m, t = 700 s
    for all 6 variants on a single axes.

    Mirrors G&R (2008) Fig. 4 but with z on the vertical axis (correct orientation).
    The x = 500 m column is the centre of the domain, where the bubble rises.
    """
    # Colour/style for each variant — distinct and accessible
    STYLES = [
        {"color": "#1a1a1a", "lw": 2.2, "ls": "-",  "zorder": 10, "label": None},  # IDEAL
        {"color": "#d62728", "lw": 1.6, "ls": "--", "zorder": 6,  "label": None},  # nabla2
        {"color": "#ff7f0e", "lw": 1.6, "ls": "-.", "zorder": 7,  "label": None},  # nabla4
        {"color": "#2ca02c", "lw": 1.6, "ls": ":",  "zorder": 8,  "label": None},  # nabla8
        {"color": "#9467bd", "lw": 1.6, "ls": (0, (4, 1.5)), "zorder": 9, "label": None},  # Shapiro
        {"color": "#17becf", "lw": 1.8, "ls": (0, (6, 2)), "zorder": 11, "label": None},  # IDEAL tiny
    ]

    # Grid for z axis
    nz = int(round(LZ / dx))
    z_km = (np.arange(nz) + 0.5) * dx / 1000.0  # cell centres in km

    # x index nearest to 500 m
    nx = int(round(LX / dx))
    x_centres = (np.arange(nx) + 0.5) * dx
    ix = int(np.argmin(np.abs(x_centres - X_C)))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for i, (snaps, label, style) in enumerate(zip(all_snaps, variant_labels, STYLES)):
        th = snaps.get(T_END)
        if th is None:
            th = snaps[max(snaps.keys())]
        profile = th[:, ix]   # shape (nz,)
        ax.plot(z_km, profile, label=label,
                color=style["color"], lw=style["lw"],
                ls=style["ls"], zorder=style["zorder"])

    ax.set_xlabel("z [km]", fontsize=12)
    ax.set_ylabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_title(
        f"Vertical profile of $\\theta'$ at $x = 500$ m,  $t = {int(T_END)}$ s\n"
        f"G&R Case 2  (dx = {dx} m)",
        fontsize=11,
    )
    ax.set_xlim(0, LZ / 1000)
    ax.set_ylim(bottom=-0.02)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color="#ddd", lw=0.6, zorder=0)
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9,
              edgecolor="#ccc", handlelength=2.8)

    fig.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="G&R Case 2 diffusion comparison — evolution grid + vertical profile")
    parser.add_argument("--dx", type=float, default=20.0,
                        help="Grid spacing [m] (default 20)")
    parser.add_argument("--shapiro-period", type=float, default=30.0,
                        help="Shapiro filter interval in simulation seconds "
                             "(default 30; use 0 for every step)")
    args = parser.parse_args()

    dx = args.dx
    dt = CFL * dx / CS
    dt_tiny = dt / 5.0
    sp = args.shapiro_period if args.shapiro_period > 0 else None

    k2 = _kappa(2, dx, dt)
    k4 = _kappa(4, dx, dt)
    k8 = _kappa(8, dx, dt)

    print(f"\n{'='*65}")
    print(f"  G&R Case 2 diffusion comparison")
    print(f"  dx={dx} m   dt={dt:.4f} s   dt_tiny={dt_tiny:.5f} s")
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
        (f"Shapiro (every {sp if sp else 'step'} s)",
         dt,  {"_shapiro": True}),
        ("IDEAL tiny dt (ref)",
         dt_tiny, {}),
    ]

    all_snaps = []
    all_stats = []
    labels    = []
    for label, vdt, gp in variants:
        print(f"Running: {label}", flush=True)
        snaps, stats = run_variant(label, dx, vdt, dict(gp), shapiro_period=sp)
        stats["dt"] = vdt
        all_snaps.append(snaps)
        all_stats.append(stats)
        labels.append(label)

    out_evo = os.path.join(OUT_DIR, f"diffcomp_gr_dx{int(dx)}m.png")
    plot_evolution_grid(all_snaps, labels, dx, out_evo)
    print_summary_table(labels, all_stats, dx)

    out_vp = os.path.join(OUT_DIR, f"diffcomp_gr_vprofile_dx{int(dx)}m.png")
    plot_vertical_profile(all_snaps, labels, dx, out_vp)


if __name__ == "__main__":
    main()
