"""
gr_si_diffusion_comparison.py
==============================
G&R Case 2: Rising Thermal Bubble — diffusion strategy comparison for SI/SI2.

Same plots as gr_diffusion_comparison.py (RK4) but variant runs use SI or
SI2 (--scheme flag), both at dt=1 s.  Variants:

  1. IDEAL  (no diffusion) — SI blows up at ~600-700 s; SI2 runs clean
  2. nabla2 — + ∇²  (κ₂ = 3 m²/s)
  3. nabla4 — + ∇⁴  (κ₄ = 200 m⁴/s)
  4. nabla8 — + ∇⁸  (κ₈ = 20000 m⁸/s)
  5. Shapiro — + Shapiro 1-2-1 filter every 30 s
  6. Reference — RK4 tiny dt (dt = 0.01 s)

Diffusion is applied as a single-level forward-Euler correction OUTSIDE the
implicit/leapfrog solve for BOTH schemes (integrators._apply_diffusion_correction
for SI2; folded into the explicit N term for SI — mathematically the same
kind of single-level update either way). Folding diffusion into SI2's
leapfrog N(q^n) term directly (the original approach) is unconditionally
unstable for any kappa > 0 — see Durran (2010) / Jablonowski & Williamson
(2011) on the leapfrog computational mode for diffusive (real-eigenvalue)
terms. With the split-step correction, SI and SI2 share the same stability
bound and the same kappa targets:

Stability limits at dx=10 m, dt=1 s (single-level forward-Euler):
  ∇²:  κ₂ ≤ Δx²/(4Δt) = 25 m²/s          → use 3 m²/s
  ∇⁴:  κ₄ ≤ Δx⁴/(32Δt) ≈ 312 m⁴/s        → use 200 m⁴/s
  ∇⁸:  κ₈ ≤ Δx⁸/(2×(8/Δx²)⁴×Δt) ≈ 48k   → use 20000 m⁸/s

USAGE
-----
  python experiments/gr_si_diffusion_comparison.py --scheme SI
  python experiments/gr_si_diffusion_comparison.py --scheme SI2
  python experiments/gr_si_diffusion_comparison.py --scheme SI2 --dx 20
"""

import argparse
import os
import sys
import time as wall_time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

# Guard against non-UTF8 console codepages (e.g. cp1252 on some Windows
# shells) crashing on the unicode symbols (nabla, superscripts) in prints.
if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

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
from integrators import step, shapiro_filter, robert_asselin_filter_time

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# G&R Case 2 parameters
# ---------------------------------------------------------------------------
LX, LZ  = 1000.0, 1000.0
THETA_C = 0.5
R_C     = 250.0
X_C     = 500.0
Z_C     = 350.0
T_END   = 700.0
SNAP_T  = [0, 100, 200, 300, 400, 500, 600, 700]
CLEV    = np.arange(0.05, 0.526, 0.025)

SI_DT       = 1.0      # s — SI large-step dt
CS          = 347.0    # m/s (speed of sound, used for acoustic CFL info)

# ---------------------------------------------------------------------------
# Kappa values (stability, safety factor 0.7)
# ---------------------------------------------------------------------------
# ∇² max eigenvalue in 2D: 8/dx²
# ∇⁴ max eigenvalue in 2D: (8/dx²)²
# ∇⁸ max eigenvalue in 2D: (8/dx²)⁴
#
# Both SI and SI2 now apply diffusion as a single-level forward-Euler
# correction OUTSIDE the implicit/leapfrog recurrence (see
# integrators._apply_diffusion_correction — folding it into SI2's leapfrog
# N(q^n) term is unconditionally unstable regardless of kappa or dt, per
# Durran 2010 / Jablonowski & Williamson 2011). Because the correction is a
# plain 2-level forward-Euler update over a single dt for BOTH schemes, the
# same CFL-type bound applies to both — no halving needed for SI2 anymore.
#   κ × eigenvalue × dt ≤ 2
def _si_kappa(order, dx, scheme="SI", dt=None):
    """Scheme-safe kappa for given order and grid spacing."""
    if dt is None:
        dt = SI_DT
    eig_max = (8.0 / dx**2) ** (order // 2)
    kappa_max = 0.7 * 2.0 / (eig_max * dt)
    # Reference target values at dx=10m (same for SI and SI2 — see note above)
    targets = {2: 3.0, 4: 200.0, 8: 20000.0}
    return min(targets[order] * (dx / 10.0)**order, kappa_max)

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
# Run one SI variant
# ---------------------------------------------------------------------------
def run_si_variant(label, dx, grid_params, snap_times=SNAP_T,
                   shapiro_period=30.0, scheme="SI", t_end=None, dt=None):
    """
    Run G&R Case 2 with SI or SI2 scheme at the given dt (defaults to the
    module-level SI_DT=1.0s if not passed).

    SI:  forward-Euler N, 1st order — blows up without diffusion/filter.
    SI2: leapfrog N + Robert-Asselin filter, 2nd order — stable without diffusion.
         Bootstrap step 0 uses SI automatically.
    Handles blow-up gracefully: stops at first NaN/Inf and saves last valid snapshot.

    t_end defaults to the module-level T_END (700s, the real benchmark);
    overriding it is only intended for quick pipeline smoke-tests, not for
    real comparisons (the plotting functions still assume T_END).
    """
    if t_end is None:
        t_end = T_END
    if dt is None:
        dt = SI_DT

    params = {"Lx": LX, "Lz": LZ, "dx": dx, "dz": dx, **grid_params}
    grid   = Grid(params)
    state  = make_ic(grid)

    use_shapiro = grid_params.pop("_shapiro", False)
    if not shapiro_period or shapiro_period <= dt:
        shapiro_every = 1
    else:
        shapiro_every = max(1, int(round(shapiro_period / dt)))

    snaps        = {}
    nstep_total  = int(round(t_end / dt))
    dt_exact     = t_end / nstep_total
    blowup_t     = None
    state_old    = None   # q^{n-1} for SI2 leapfrog

    target_steps = {}
    for ts in snap_times:
        if ts == 0:
            snaps[0] = {"theta": state["theta"].copy(),
                        "u":     state["u"].copy(),
                        "w":     state["w"].copy()}
        else:
            idx = int(round(ts / dt_exact))
            target_steps[min(idx, nstep_total)] = ts

    t0 = wall_time.perf_counter()
    for n in range(nstep_total):
        try:
            state_new, state_prev, _ = step(state, grid, dt_exact,
                                            scheme=scheme,
                                            state_old=state_old)
        except Exception as e:
            blowup_t = (n + 1) * dt_exact
            print(f"  {label}: exception at t={blowup_t:.0f}s — {e}", flush=True)
            break

        # Check for blow-up
        if not np.isfinite(state_new["theta"]).all() or \
           np.abs(state_new["theta"]).max() > 50.0:
            blowup_t = (n + 1) * dt_exact
            print(f"  {label}: blow-up detected at t={blowup_t:.0f}s", flush=True)
            break

        # SI2/SI2LU: Robert-Asselin filter on q^n before advancing (damps computational mode)
        if scheme in ("SI2", "SI2LU") and state_old is not None:
            state_filtered = robert_asselin_filter_time(state_old, state, state_new)
        else:
            state_filtered = state

        # Advance: q^{n-1} ← filtered q^n,  q^n ← q^{n+1}
        state_old = state_filtered
        state     = state_new

        if use_shapiro and (n + 1) % shapiro_every == 0:
            state = shapiro_filter(state, grid)

        if (n + 1) in target_steps:
            snaps[target_steps[n + 1]] = {
                "theta": state["theta"].copy(),
                "u":     state["u"].copy(),
                "w":     state["w"].copy(),
            }

    elapsed = wall_time.perf_counter() - t0
    th = state["theta"]
    w  = state["w"]
    stats = {
        "theta_max": float(np.nanmax(th)),
        "theta_min": float(np.nanmin(th)),
        "w_max":     float(np.nanmax(w)),
        "w_min":     float(np.nanmin(w)),
        "wall":      elapsed,
        "dt":        dt_exact,
        "nstep":     nstep_total,
        "blowup_t":  blowup_t,
    }
    blow_str = f"  *** BLOW-UP at t={blowup_t:.0f}s ***" if blowup_t else ""
    print(f"  {label:<40} theta_max={stats['theta_max']:.3f} K  "
          f"w_max={stats['w_max']:.3f} m/s  wall={elapsed:.1f}s{blow_str}", flush=True)
    return snaps, stats


def _run_variant_worker(payload):
    """
    Module-level (picklable) wrapper so each variant can run in its own
    process via ProcessPoolExecutor -- the variants are fully independent
    (different diffusion settings on the same IC), so there is no reason to
    run them sequentially.
    """
    label, dx, gp, shapiro_period, scheme, t_end, dt = payload
    snaps, stats = run_si_variant(label, dx, dict(gp),
                                  shapiro_period=shapiro_period,
                                  scheme=scheme, t_end=t_end, dt=dt)
    return label, snaps, stats


# ---------------------------------------------------------------------------
# Evolution grid figure  (reused from gr_diffusion_comparison)
# ---------------------------------------------------------------------------
def plot_evolution_grid(all_snaps, variant_labels, dx, out_path, scheme="SI", dt=None):
    if dt is None:
        dt = SI_DT
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

    # squeeze=False guarantees a 2D axes array even when n_rows==1 (e.g.
    # running with a single --variants selection) -- see plot_final_snapshot
    # for the same fix and why it's needed.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(
        left   = PAD_L / fig_w,
        right  = (PAD_L + n_cols * CELL_W) / fig_w,
        top    = 1.0 - PAD_T / fig_h,
        bottom = PAD_B / fig_h,
        hspace = 0.04, wspace = 0.04,
    )

    cmap       = "RdYlBu_r"
    vmin, vmax = CLEV[0], CLEV[-1]
    extent     = [0, LX / 1000, 0, LZ / 1000]

    for row, (snaps, label) in enumerate(zip(all_snaps, variant_labels)):
        for col, ts in enumerate(SNAP_T):
            ax = axes[row][col]
            ax.set_facecolor("#0a1628")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334")
                spine.set_linewidth(0.5)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

            snap = snaps.get(ts)
            if snap is not None:
                th  = snap["theta"]
                u_f = snap["u"]
                w_f = snap["w"]
                disp = gaussian_filter(th, sigma=0.9) if SMOOTH else th

                ax.imshow(disp, origin="lower", extent=extent,
                          aspect="auto", cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="bicubic")

                x_km = np.linspace(0, LX / 1000, th.shape[1])
                z_km = np.linspace(0, LZ / 1000, th.shape[0])
                X, Z = np.meshgrid(x_km, z_km)
                ax.contour(X, Z, disp, levels=CLEV,
                           colors="white", linewidths=0.35, alpha=0.55)

                skip = max(1, int(round(80.0 / dx)))
                ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                          u_f[::skip, ::skip], w_f[::skip, ::skip],
                          color="white", alpha=0.55,
                          scale=35, width=0.004,
                          headwidth=3, headlength=4)
            else:
                # Blow-up: show dark panel with label
                ax.text(0.5, 0.5, "blow-up", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7,
                        color="#ff6666", fontweight="bold")

            ax.set_xlim(0, LX / 1000)
            ax.set_ylim(0, LZ / 1000)

            if row == 0:
                ax.set_title(f"t = {ts} s", fontsize=8.5,
                             fontweight="bold", color="#222", pad=4)
            if row == n_rows - 1:
                ax.tick_params(labelbottom=True)
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax.tick_params(axis="x", labelsize=6.5)
                if col == 0:
                    ax.set_xlabel("x [km]", fontsize=7, labelpad=2)
            if col == 0:
                ax.tick_params(labelleft=True)
                ax.set_ylabel("z [km]", fontsize=7, labelpad=2)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax.tick_params(axis="y", labelsize=6.5)
                ax.text(-0.38, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=8.5, fontweight="bold", color="#111",
                        ha="right", va="center", rotation=0)

    # Colorbar
    cbar_left   = (PAD_L + n_cols * CELL_W + 0.12) / fig_w
    cbar_bottom = PAD_B / fig_h + 0.04
    cbar_width  = 0.20 / fig_w
    cbar_height = 1.0 - (PAD_T + PAD_B) / fig_h - 0.06
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\theta'$ [K]", fontsize=10, labelpad=6)
    cbar.ax.tick_params(labelsize=8.5)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        f"G&R Case 2 — Rising Thermal Bubble  [{scheme} scheme, dt={dt:g} s]"
        f"   (dx = {dx} m,  t_end = {T_END:.0f} s)",
        fontsize=11, fontweight="bold", color="#111",
        y=1.0 - 0.08 / fig_h,
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Final snapshot figure
# ---------------------------------------------------------------------------
def plot_final_snapshot(all_snaps, variant_labels, dx, out_path, scheme="SI", dt=None):
    if dt is None:
        dt = SI_DT
    try:
        from scipy.ndimage import gaussian_filter
        SMOOTH = True
    except ImportError:
        SMOOTH = False

    n_vars = len(all_snaps)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols

    CELL  = 3.6
    PAD_L = 0.55
    PAD_R = 0.80
    PAD_T = 0.55
    PAD_B = 0.45

    fig_w = PAD_L + n_cols * CELL + PAD_R
    fig_h = PAD_T + n_rows * CELL + PAD_B

    # squeeze=False guarantees a 2D axes array even when n_rows==1 or
    # n_cols==1 (e.g. running with only 2-3 variants via --variants) --
    # without it, plt.subplots collapses to a 1D array or a bare Axes
    # object and axes[row][col] below fails.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, squeeze=False)
    plt.subplots_adjust(
        left   = PAD_L / fig_w,
        right  = (PAD_L + n_cols * CELL) / fig_w,
        top    = 1.0 - PAD_T / fig_h,
        bottom = PAD_B / fig_h,
        hspace = 0.08, wspace = 0.06,
    )

    cmap       = "RdYlBu_r"
    vmin, vmax = CLEV[0], CLEV[-1]
    extent     = [0, LX / 1000, 0, LZ / 1000]

    for idx, (snaps, label) in enumerate(zip(all_snaps, variant_labels)):
        row = idx // n_cols
        col = idx  % n_cols
        ax  = axes[row][col]

        ax.set_facecolor("#0a1628")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334")
            spine.set_linewidth(0.6)

        # Use last available snapshot (handles blow-up case)
        snap_t = T_END
        snap = snaps.get(T_END)
        if snap is None and snaps:
            snap_t = max(snaps.keys())
            snap = snaps[snap_t]

        if snap is not None:
            th  = snap["theta"]
            u_f = snap["u"]
            w_f = snap["w"]
            disp = gaussian_filter(th, sigma=0.9) if SMOOTH else th

            ax.imshow(disp, origin="lower", extent=extent,
                      aspect="auto", cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="bicubic")

            x_km = np.linspace(0, LX / 1000, th.shape[1])
            z_km = np.linspace(0, LZ / 1000, th.shape[0])
            X, Z = np.meshgrid(x_km, z_km)
            ax.contour(X, Z, disp, levels=CLEV,
                       colors="white", linewidths=0.4, alpha=0.55)

            skip = max(1, int(round(80.0 / dx)))
            ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                      u_f[::skip, ::skip], w_f[::skip, ::skip],
                      color="white", alpha=0.6,
                      scale=35, width=0.004,
                      headwidth=3, headlength=4)

        ax.set_xlim(0, LX / 1000)
        ax.set_ylim(0, LZ / 1000)

        if row == n_rows - 1:
            ax.tick_params(labelbottom=True)
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("x [km]", fontsize=9, labelpad=3)
        if col == 0:
            ax.tick_params(labelleft=True)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.tick_params(axis="y", labelsize=8)
            ax.set_ylabel("z [km]", fontsize=9, labelpad=3)

        panel_title = label if snap_t == T_END else f"{label}  [blew up, shown t={snap_t:.0f}s]"
        ax.set_title(panel_title, fontsize=9.5, fontweight="bold",
                     color="#111", pad=5)

    # Colorbar
    cbar_left   = (PAD_L + n_cols * CELL + 0.12) / fig_w
    cbar_bottom = PAD_B / fig_h + 0.04
    cbar_width  = 0.22 / fig_w
    cbar_height = 1.0 - (PAD_T + PAD_B) / fig_h - 0.06
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\theta'$ [K]", fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        f"G&R Case 2 — Rising Thermal Bubble at t = {int(T_END)} s"
        f"  [{scheme} scheme, dt={dt:g} s]   (dx = {dx} m)",
        fontsize=12, fontweight="bold", color="#111",
        y=1.0 - 0.10 / fig_h,
    )
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Vertical profile
# ---------------------------------------------------------------------------
def plot_vertical_profile(all_snaps, variant_labels, dx, out_path, scheme="SI", dt=None):
    if dt is None:
        dt = SI_DT
    STYLES = [
        {"color": "#1a1a1a", "lw": 2.2, "ls": "-",           "zorder": 10},
        {"color": "#d62728", "lw": 1.6, "ls": "--",          "zorder": 6},
        {"color": "#ff7f0e", "lw": 1.6, "ls": "-.",          "zorder": 7},
        {"color": "#2ca02c", "lw": 1.6, "ls": ":",           "zorder": 8},
        {"color": "#9467bd", "lw": 1.6, "ls": (0, (4, 1.5)), "zorder": 9},
        {"color": "#17becf", "lw": 1.8, "ls": (0, (6, 2)),   "zorder": 11},
    ]

    nz   = int(round(LZ / dx))
    z_km = (np.arange(nz) + 0.5) * dx / 1000.0
    nx   = int(round(LX / dx))
    x_c  = (np.arange(nx) + 0.5) * dx
    ix   = int(np.argmin(np.abs(x_c - X_C)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    any_early = False
    for i, (snaps, label, style) in enumerate(zip(all_snaps, variant_labels, STYLES)):
        snap_t = T_END
        snap = snaps.get(T_END)
        if snap is None and snaps:
            snap_t = max(snaps.keys())
            snap = snaps[snap_t]
        if snap is None:
            continue
        # Blown-up variants: this is the last valid snapshot, not t=T_END --
        # say so explicitly in the legend rather than letting the shared
        # "t=700s" title imply every curve reached the full simulation time.
        plot_label = label
        if snap_t != T_END:
            plot_label = f"{label}  [blew up, shown t={snap_t:.0f}s]"
            any_early = True
        th = snap["theta"] if isinstance(snap, dict) else snap
        ax.plot(th[:, ix], z_km, label=plot_label,
                color=style["color"], lw=style["lw"],
                ls=style["ls"], zorder=style["zorder"])

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z [km]", fontsize=12)
    end_note = "\n(some variants blew up before t=700 s -- see legend)" if any_early else ""
    ax.set_title(
        f"Vertical profile of $\\theta'$ at $x = 500$ m,  $t = {int(T_END)}$ s{end_note}\n"
        f"G&R Case 2  [{scheme} scheme, dt={dt:g} s]  (dx = {dx} m)",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 0.70)
    ax.set_ylim(0.45, 1.02)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
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
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(labels, all_stats, dx, scheme="SI"):
    W   = 95
    sep = "+" + "-"*42 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*9 + "+"
    hdr = (f"| {'Variant':<40} | {'dt(s)':>8} | {'th_max(K)':>8} "
           f"| {'th_min(K)':>8} | {'wmax(m/s)':>8} | {'Wall(s)':>7} |")

    print("\n" + "="*W)
    print(f"  {scheme} DIFFUSION COMPARISON  —  G&R Case 2, dx={dx} m, t=700 s")
    print("="*W)
    print(sep); print(hdr); print(sep)
    for label, s in zip(labels, all_stats):
        blow = f" [BLOWUP@{s['blowup_t']:.0f}s]" if s.get("blowup_t") else ""
        print(f"| {label:<40} | {s['dt']:>8.4f} | {s['theta_max']:>8.4f} "
              f"| {s['theta_min']:>8.4f} | {s['w_max']:>8.4f} | {s['wall']:>7.1f} |{blow}")
    print(sep); print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="G&R Case 2 diffusion comparison — SI or SI2 scheme")
    parser.add_argument("--dx", type=float, default=10.0,
                        help="Grid spacing [m] (default 10)")
    parser.add_argument("--scheme", choices=["SI", "SI2", "SI2LU"], default="SI",
                        help="Time scheme: SI (1st order), SI2 (2nd order leapfrog, "
                             "GMRES), or SI2LU (same math as SI2, solved via a "
                             "sparse LU factorization built once and reused every "
                             "step instead of GMRES -- ~100-200x faster). Default SI.")
    parser.add_argument("--shapiro-period", type=float, default=30.0,
                        help="Shapiro filter interval [s] (default 30)")
    parser.add_argument("--variants", default="ideal,nabla2,nabla4,nabla8,shapiro",
                        help="Comma-separated subset of {ideal,nabla2,nabla4,nabla8,shapiro} "
                             "to run (default: all five). E.g. --variants nabla2,nabla4,nabla8")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: one per selected variant, "
                             "capped at CPU count). Variants are fully independent, so they "
                             "run concurrently rather than one after another.")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override SI/SI2/SI2LU time step [s] (default: module "
                             "SI_DT=1.0s). Also rescales the auto-capped kappa targets "
                             "for the new dt.")
    parser.add_argument("--replot-only", metavar="PATH", default=None,
                        help="Skip the simulation entirely and regenerate all three figures "
                             "from a previously-saved results cache (see the 'Cached results' "
                             "path printed after a normal run). Useful if a plotting bug ever "
                             "crashes the script after an expensive run -- the simulation "
                             "results are cached BEFORE plotting specifically so they survive "
                             "a plotting-side crash.")
    args = parser.parse_args()

    if args.replot_only:
        import pickle
        with open(args.replot_only, "rb") as f:
            cache = pickle.load(f)
        all_labels, all_stats, all_snaps = cache["labels"], cache["stats"], cache["snaps"]
        dx, scheme = cache["dx"], cache["scheme"]
        dt_cached = cache.get("dt")   # None for caches saved before this field existed
        print(f"  Loaded cached results from {args.replot_only} "
              f"({len(all_labels)} variant(s), scheme={scheme}, dx={dx}m, "
              f"dt={dt_cached if dt_cached is not None else SI_DT}s)")
        print_summary_table(all_labels, all_stats, dx, scheme=scheme)
        dt_tag = f"_dt{dt_cached:g}".replace(".", "p") if dt_cached is not None else ""
        tag = f"{scheme.lower()}_dx{int(dx)}m{dt_tag}"
        plot_evolution_grid(all_snaps, all_labels, dx,
                            os.path.join(OUT_DIR, f"si_diffcomp_evolution_{tag}.png"),
                            scheme=scheme, dt=dt_cached)
        plot_final_snapshot(all_snaps, all_labels, dx,
                            os.path.join(OUT_DIR, f"si_diffcomp_final_{tag}.png"),
                            scheme=scheme, dt=dt_cached)
        plot_vertical_profile(all_snaps, all_labels, dx,
                              os.path.join(OUT_DIR, f"si_diffcomp_vprofile_{tag}.png"),
                              scheme=scheme, dt=dt_cached)
        return

    dx     = args.dx
    scheme = args.scheme
    dt_arg = args.dt   # None -> run_si_variant/_si_kappa fall back to SI_DT
    k2 = _si_kappa(2, dx, scheme, dt=dt_arg)
    k4 = _si_kappa(4, dx, scheme, dt=dt_arg)
    k8 = _si_kappa(8, dx, scheme, dt=dt_arg)
    sp = args.shapiro_period

    print(f"\n{'='*65}")
    print(f"  G&R Case 2 — {scheme} diffusion comparison")
    print(f"  dx={dx} m   dt={dt_arg if dt_arg is not None else SI_DT} s")
    print(f"  kappa2={k2:.2f} m²/s   kappa4={k4:.1f} m⁴/s   kappa8={k8:.2e} m⁸/s")
    print(f"  Shapiro every {sp:.0f} s")
    if scheme in ("SI2", "SI2LU"):
        print(f"  Robert-Asselin filter: alpha=0.1 (applied every step)")
        if scheme == "SI2LU":
            print(f"  Solver: sparse LU factorized once, reused every step "
                  f"(not GMRES) -- same math as SI2, ~100-200x faster")
    print(f"{'='*65}\n")

    all_variants = [
        ("ideal",   "IDEAL  (no diffusion)",       {}),
        ("nabla2",  f"nabla2  (k2={k2:.2g} m2/s)", {"diffusion_coeff": k2, "diffusion_order": 2}),
        ("nabla4",  f"nabla4  (k4={k4:.2g} m4/s)", {"diffusion_coeff": k4, "diffusion_order": 4}),
        ("nabla8",  f"nabla8  (k8={k8:.2e} m8/s)", {"diffusion_coeff": k8, "diffusion_order": 8}),
        ("shapiro", f"Shapiro (every {sp:.0f} s)",  {"_shapiro": True}),
    ]
    selected = {k.strip().lower() for k in args.variants.split(",") if k.strip()}
    variants = [(label, gp) for key, label, gp in all_variants if key in selected]
    if not variants:
        raise ValueError(f"--variants={args.variants!r} matched none of "
                          f"{[k for k,_,_ in all_variants]}")

    n_workers = args.workers or min(len(variants), os.cpu_count() or 1)

    # Avoid BLAS thread oversubscription: each worker process runs its own
    # GMRES-heavy simulation. Without pinning BLAS to 1 thread per process,
    # N processes each spawning M BLAS threads can massively oversubscribe
    # the CPU and make "parallel" slower than the original sequential runs.
    # Must be set before numpy is imported in each child, so set it here in
    # the parent BEFORE spawning -- child processes inherit this environment
    # at spawn time and then do their own fresh numpy import.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    print(f"  Running {len(variants)} variant(s) across {n_workers} "
          f"worker process(es) in parallel...\n")

    results_by_label = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_variant_worker,
                            (label, dx, gp, sp, scheme, None, dt_arg)): label
            for label, gp in variants
        }
        for fut in as_completed(futures):
            label, snaps, stats = fut.result()
            results_by_label[label] = (snaps, stats)

    # Reassemble in the original (deterministic) order regardless of which
    # process finished first, so the plots' row/legend order stays stable.
    all_snaps  = []
    all_stats  = []
    all_labels = []
    for label, gp in variants:
        snaps, stats = results_by_label[label]
        all_snaps.append(snaps)
        all_stats.append(stats)
        all_labels.append(label)

    print_summary_table(all_labels, all_stats, dx, scheme=scheme)

    # Cache the (expensive!) simulation results to disk BEFORE any plotting
    # is attempted. A bug in a plotting function should never be able to
    # destroy hours of GMRES compute -- if plotting crashes below, this
    # cache lets you regenerate the figures via --replot-only <path>
    # without re-running the simulation.
    import pickle
    dt_tag = f"_dt{dt_arg:g}".replace(".", "p") if dt_arg is not None else ""
    tag = f"{scheme.lower()}_dx{int(dx)}m{dt_tag}"
    results_dir  = os.path.join("output", "results")
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, f"si_diffcomp_{tag}_cache.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"labels": all_labels, "stats": all_stats,
                    "snaps": all_snaps, "dx": dx, "scheme": scheme, "dt": dt_arg}, f)
    print(f"  Cached results (pre-plotting) -> {cache_path}\n"
          f"  If plotting fails below, regenerate figures with:\n"
          f"    python {os.path.basename(__file__)} --replot-only {cache_path}\n")

    plot_evolution_grid(all_snaps, all_labels, dx,
                        os.path.join(OUT_DIR, f"si_diffcomp_evolution_{tag}.png"),
                        scheme=scheme, dt=dt_arg)
    plot_final_snapshot(all_snaps, all_labels, dx,
                        os.path.join(OUT_DIR, f"si_diffcomp_final_{tag}.png"),
                        scheme=scheme, dt=dt_arg)
    plot_vertical_profile(all_snaps, all_labels, dx,
                          os.path.join(OUT_DIR, f"si_diffcomp_vprofile_{tag}.png"),
                          scheme=scheme, dt=dt_arg)


if __name__ == "__main__":
    main()
