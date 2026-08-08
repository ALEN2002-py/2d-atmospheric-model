"""
pc_si_diffusion_comparison.py
==============================
P&C (2022) Experiment 1: Convective Bubble -- diffusion strategy comparison
for SI/SI2, on the paper's 5km domain with the flat-top+Gaussian bubble IC.

This is the P&C analogue of gr_si_diffusion_comparison.py (which mirrors
diffusion_comparison.py's RK4-based comparison but for SI/SI2). Until now
there was no SI/SI2 diffusion sweep for P&C at all -- pc_diffusion_comparison.py
and pc_kappa_sweep.py are both RK4-only.

Variants:
  1. IDEAL   (no diffusion) -- baseline; SI/SI2 stability without any filter
  2. nabla2  -- + Laplacian       kappa2 * nabla^2
  3. nabla4  -- + biharmonic     -kappa4 * nabla^4
  4. nabla8  -- + octaharmonic   -kappa8 * nabla^8
  5. Shapiro -- + Shapiro 1-2-1 filter every ~30 s

Diffusion is applied as a single-level forward-Euler correction OUTSIDE the
implicit/leapfrog solve for BOTH SI and SI2 (see integrators.py:
_apply_diffusion_correction and _semi_implicit's inline kappa handling) --
folding it into SI2's leapfrog N(q^n) term directly is unconditionally
unstable for any kappa > 0 (Durran 2010; Jablonowski & Williamson 2011).
With the split-step correction, SI and SI2 share the same stability bound:
  kappa * eigenvalue_max * dt <= 2   (single-level forward-Euler)

kappa reference targets (at dx=20m, the paper resolution) are chosen via the
same damping-timescale argument used throughout this codebase (README Sec 8):
  tau = 1 / (kappa * (pi/dx)^n)   targeting tau ~ 50s at the 2*dx wave.
These are a first-pass, physically-motivated starting point (same
methodology as the G&R script), not yet empirically tuned against real
P&C runs -- expect to refine after seeing the first comparison.

USAGE
-----
  python experiments/pc_si_diffusion_comparison.py --scheme SI
  python experiments/pc_si_diffusion_comparison.py --scheme SI2
  python experiments/pc_si_diffusion_comparison.py --scheme SI2 --dx 40 --variants nabla2,nabla4,nabla8
"""

import argparse
import math
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
# P&C Exp 1 parameters (Section 6.1)
# ---------------------------------------------------------------------------
LX, LZ  = 5000.0, 5000.0
AT      = 0.5       # K   bubble amplitude
A_RAD   = 400.0     # m   flat-top radius (paper: 20*delta at delta=20m)
SIGMA   = 100.0     # m   Gaussian edge width (paper: 5*delta)
X0      = 2500.0    # m   bubble centre x
Z0      = 700.0     # m   bubble centre z

# Full paper duration (1800s), unlike pc_diffusion_comparison.py's RK4-tuned
# 1200s (chosen there as roughly where the pure-RK4 IDEAL blows up). Here we
# want to see whether SI/SI2 + diffusion can sustain the FULL paper benchmark.
T_END   = 1800.0
SNAP_T  = [0, 300, 600, 900, 1200, 1500, 1800]
CLEV    = np.arange(0.02, 0.501, 0.02)

SI_DT = 5.0      # s -- matches pc_exp1_benchmark.py's SI/SI2 default dt
CS    = 347.0    # m/s (speed of sound, for acoustic CFL info only)

_G    = 9.81
_TBAR = 300.0
_B0   = _G * AT / _TBAR
_L    = 2.0 * A_RAD
T_BUOY  = math.sqrt(_L / _B0)     # ~ 220 s
W_SCALE = math.sqrt(_B0 * _L)     # ~ 3.6 m/s

# ---------------------------------------------------------------------------
# Kappa values (stability, safety factor 0.7) -- same functional form as
# gr_si_diffusion_comparison.py._si_kappa, but with P&C's own reference
# targets (larger bubble scale -> gentler damping needed) and P&C's own
# SI_DT (5s, not 1s).
# ---------------------------------------------------------------------------
_DX_REF = 20.0   # paper resolution -- reference targets are quoted at this dx

def _damping_kappa(order, dx, tau_target=50.0):
    """kappa such that the 2*dx-wave damping timescale is tau_target seconds:
    tau = 1 / (kappa * (pi/dx)^order)  =>  kappa = 1 / (tau_target * (pi/dx)^order)
    """
    return 1.0 / (tau_target * (math.pi / dx) ** order)

# Reference targets at dx=20m (paper resolution), tau~50s -- see _damping_kappa.
_K2_REF = _damping_kappa(2, _DX_REF)
_K4_REF = _damping_kappa(4, _DX_REF)
_K8_REF = _damping_kappa(8, _DX_REF)

def _si_kappa(order, dx, dt=None, scale=1.0):
    """Scheme-safe kappa for given order, grid spacing, and dt (SI and SI2
    share the same bound now that diffusion is a single-level forward-Euler
    correction for both -- see module docstring). dt defaults to SI_DT but
    MUST be passed explicitly whenever --dt overrides the default, since the
    stability cap kappa_max = 0.7*2/(eig_max*dt) depends on it directly.

    scale: multiplier on the tau=50s-based target, BEFORE the stability cap
    is applied. Added after discovering (first real dx=40m SI run, all 5
    variants blew up) that this formula's default targets are far weaker
    than the already-validated RK4 reference values at the same resolution:
    kappa4 ~8x weaker, kappa8 ~1580x weaker (kappa2 was comparable, ~0.8x).
    Unlike the G&R SI kappa values (tuned and validated over many real runs
    this project phase), this P&C formula had never been run for real before
    that test, so treat it as a starting point to be empirically re-tuned,
    not a trusted value -- use --kappa-scale to sweep strength cheaply
    (blown-up runs finish in minutes) before committing to the expensive
    paper-resolution dx=20m run.
    """
    if dt is None:
        dt = SI_DT
    eig_max   = (8.0 / dx**2) ** (order // 2)
    kappa_max = 0.7 * 2.0 / (eig_max * dt)
    ref       = {2: _K2_REF, 4: _K4_REF, 8: _K8_REF}[order]
    target    = ref * (dx / _DX_REF) ** order * scale
    return min(target, kappa_max)

# ---------------------------------------------------------------------------
# Initial condition -- P&C flat-top + Gaussian-edge bubble
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
# Run one SI/SI2 variant
# ---------------------------------------------------------------------------
def run_si_variant(label, dx, grid_params, snap_times=SNAP_T,
                   shapiro_period=30.0, scheme="SI", t_end=None, dt=None):
    """
    Run P&C Exp 1 with SI or SI2 at the given dt (defaults to module-level
    SI_DT=5s if not passed -- pass explicitly whenever --dt overrides it).

    t_end defaults to the module-level T_END (1800s, the full paper
    benchmark); overriding it is only intended for quick pipeline
    smoke-tests, not for real comparisons.
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
        if ts > t_end:
            continue   # can't snapshot a time beyond this run's own t_end
        if ts == 0:
            snaps[0] = {"theta": state["theta"].copy(),
                        "u":     state["u"].copy(),
                        "w":     state["w"].copy()}
        else:
            idx = int(round(ts / dt_exact))
            target_steps[idx] = ts

    t0 = wall_time.perf_counter()
    for n in range(nstep_total):
        try:
            state_new, state_prev, _ = step(state, grid, dt_exact,
                                            scheme=scheme,
                                            state_old=state_old)
        except Exception as e:
            blowup_t = (n + 1) * dt_exact
            print(f"  {label}: exception at t={blowup_t:.0f}s -- {e}", flush=True)
            break

        # Check for blow-up (amplitude threshold well above bubble scale AT=0.5K)
        if not np.isfinite(state_new["theta"]).all() or \
           np.abs(state_new["theta"]).max() > 50.0:
            blowup_t = (n + 1) * dt_exact
            print(f"  {label}: blow-up detected at t={blowup_t:.0f}s", flush=True)
            break

        if scheme in ("SI2", "SI2LU") and state_old is not None:
            state_filtered = robert_asselin_filter_time(state_old, state, state_new)
        else:
            state_filtered = state

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
    """Module-level (picklable) wrapper so each variant runs in its own
    process via ProcessPoolExecutor -- the variants are fully independent."""
    label, dx, gp, shapiro_period, scheme, t_end, dt = payload
    snaps, stats = run_si_variant(label, dx, dict(gp),
                                  shapiro_period=shapiro_period,
                                  scheme=scheme, t_end=t_end, dt=dt)
    return label, snaps, stats


# ---------------------------------------------------------------------------
# Evolution grid figure
# ---------------------------------------------------------------------------
def plot_evolution_grid(all_snaps, variant_labels, dx, out_path, scheme="SI", t_end=None, dt=None):
    if t_end is None:
        t_end = T_END
    if dt is None:
        dt = SI_DT
    try:
        from scipy.ndimage import gaussian_filter
        SMOOTH = True
    except ImportError:
        SMOOTH = False

    snap_cols = [ts for ts in SNAP_T if ts <= t_end]
    n_rows = len(all_snaps)
    n_cols = len(snap_cols)

    PAD_L  = 1.8
    CELL_W = 1.55
    CELL_H = 1.55
    PAD_R  = 0.70
    PAD_T  = 0.65
    PAD_B  = 0.30

    fig_w = PAD_L + n_cols * CELL_W + PAD_R
    fig_h = PAD_T + n_rows * CELL_H + PAD_B

    # squeeze=False guarantees a 2D axes array even when n_rows==1 (e.g. a
    # single --variants selection) -- see pc_diffusion_comparison.py /
    # gr_si_diffusion_comparison.py for the same fix and why it's needed.
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
        for col, ts in enumerate(snap_cols):
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

                skip = max(1, int(round(320.0 / dx)))
                ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                          u_f[::skip, ::skip], w_f[::skip, ::skip],
                          color="white", alpha=0.55,
                          scale=40, width=0.004,
                          headwidth=3, headlength=4)
            else:
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
                ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
                ax.tick_params(axis="x", labelsize=6.5)
                if col == 0:
                    ax.set_xlabel("x [km]", fontsize=7, labelpad=2)
            if col == 0:
                ax.tick_params(labelleft=True)
                ax.set_ylabel("z [km]", fontsize=7, labelpad=2)
                ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
                ax.tick_params(axis="y", labelsize=6.5)
                ax.text(-0.38, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=8.5, fontweight="bold", color="#111",
                        ha="right", va="center", rotation=0)

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
        f"P&C Exp 1 -- Convective Bubble  [{scheme} scheme, dt={dt:g} s]"
        f"   (dx = {dx} m,  t_end = {t_end:.0f} s)",
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
def plot_final_snapshot(all_snaps, variant_labels, dx, out_path, scheme="SI", t_end=None, dt=None):
    if t_end is None:
        t_end = T_END
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

        snap_t = t_end
        snap = snaps.get(t_end)
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

            skip = max(1, int(round(320.0 / dx)))
            ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
                      u_f[::skip, ::skip], w_f[::skip, ::skip],
                      color="white", alpha=0.6,
                      scale=40, width=0.004,
                      headwidth=3, headlength=4)

        ax.set_xlim(0, LX / 1000)
        ax.set_ylim(0, LZ / 1000)

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

        panel_title = label if snap_t == t_end else f"{label}  [blew up, shown t={snap_t:.0f}s]"
        ax.set_title(panel_title, fontsize=9.5, fontweight="bold",
                     color="#111", pad=5)

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
        f"P&C Exp 1 -- Convective Bubble at t = {int(t_end)} s"
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
def plot_vertical_profile(all_snaps, variant_labels, dx, out_path, scheme="SI", t_end=None, dt=None):
    if t_end is None:
        t_end = T_END
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
    ix   = int(np.argmin(np.abs(x_c - X0)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    any_early = False
    for i, (snaps, label, style) in enumerate(zip(all_snaps, variant_labels, STYLES)):
        snap_t = t_end
        snap = snaps.get(t_end)
        if snap is None and snaps:
            snap_t = max(snaps.keys())
            snap = snaps[snap_t]
        if snap is None:
            continue
        plot_label = label
        if snap_t != t_end:
            plot_label = f"{label}  [blew up, shown t={snap_t:.0f}s]"
            any_early = True
        th = snap["theta"] if isinstance(snap, dict) else snap
        ax.plot(th[:, ix], z_km, label=plot_label,
                color=style["color"], lw=style["lw"],
                ls=style["ls"], zorder=style["zorder"])

    ax.set_xlabel(r"$\theta'$ [K]", fontsize=12)
    ax.set_ylabel("z [km]", fontsize=12)
    end_note = f"\n(some variants blew up before t={t_end:.0f} s -- see legend)" if any_early else ""
    ax.set_title(
        f"Vertical profile of $\\theta'$ at $x = {X0:.0f}$ m,  $t = {int(t_end)}$ s{end_note}\n"
        f"P&C Exp 1  [{scheme} scheme, dt={dt:g} s]  (dx = {dx} m)",
        fontsize=11,
    )
    ax.set_xlim(-0.05, 0.60)
    ax.set_ylim(0.0, LZ / 1000.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.grid(axis="x", color="#ddd", lw=0.6, zorder=0)
    ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="#ccc", handlelength=2.8)

    fig.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary_table(labels, all_stats, dx, scheme="SI", t_end=None):
    if t_end is None:
        t_end = T_END
    W   = 95
    sep = "+" + "-"*42 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*9 + "+"
    hdr = (f"| {'Variant':<40} | {'dt(s)':>8} | {'th_max(K)':>8} "
           f"| {'th_min(K)':>8} | {'wmax(m/s)':>8} | {'Wall(s)':>7} |")

    print("\n" + "="*W)
    print(f"  {scheme} DIFFUSION COMPARISON  --  P&C Exp 1, dx={dx} m, t={t_end:.0f} s")
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
        description="P&C Exp 1 diffusion comparison -- SI or SI2 scheme")
    parser.add_argument("--dx", type=float, default=20.0,
                        help="Grid spacing [m] (default 20, paper resolution)")
    parser.add_argument("--scheme", choices=["SI", "SI2", "SI2LU"], default="SI",
                        help="Time scheme: SI (1st order), SI2 (2nd order leapfrog, "
                             "GMRES), or SI2LU (same math as SI2, solved via a sparse "
                             "LU factorization built once and reused every step instead "
                             "of GMRES -- dramatically faster).")
    parser.add_argument("--shapiro-period", type=float, default=30.0,
                        help="Shapiro filter interval [s] (default 30)")
    parser.add_argument("--variants", default="ideal,nabla2,nabla4,nabla8,shapiro",
                        help="Comma-separated subset of {ideal,nabla2,nabla4,nabla8,shapiro} "
                             "to run (default: all five)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: one per selected "
                             "variant, capped at CPU count)")
    parser.add_argument("--replot-only", metavar="PATH", default=None,
                        help="Skip the simulation and regenerate figures from a "
                             "previously-saved results cache")
    parser.add_argument("--kappa-scale", type=float, default=1.0,
                        help="Multiplier on the diffusion targets before the stability "
                             "cap (default 1.0). The default tau=50s formula was found "
                             "to be far weaker than the already-validated RK4 reference "
                             "at the same resolution (kappa4 ~8x, kappa8 ~1580x weaker) "
                             "-- use this to sweep strength cheaply (blown-up runs finish "
                             "in minutes) before trusting a result.")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override SI/SI2 time step [s] (default: module SI_DT=5s). "
                             "run_si_variant/_si_kappa already supported this; this flag "
                             "was the missing piece that actually exposes it.")
    parser.add_argument("--t-end", type=float, default=None,
                        help="Override simulation end time [s] (default: module T_END=1800s, "
                             "the full paper duration). E.g. --t-end 1200 to stop before the "
                             "domain-top interaction seen late in the full-length run.")
    args = parser.parse_args()

    if args.replot_only:
        import pickle
        with open(args.replot_only, "rb") as f:
            cache = pickle.load(f)
        all_labels, all_stats, all_snaps = cache["labels"], cache["stats"], cache["snaps"]
        dx, scheme = cache["dx"], cache["scheme"]
        t_end_cached = cache.get("t_end")   # None for caches saved before this field existed
        dt_cached = cache.get("dt")   # None for caches saved before this field existed (falls back to SI_DT)
        print(f"  Loaded cached results from {args.replot_only} "
              f"({len(all_labels)} variant(s), scheme={scheme}, dx={dx}m, "
              f"dt={dt_cached if dt_cached is not None else SI_DT}s, "
              f"t_end={t_end_cached if t_end_cached is not None else T_END}s)")
        print_summary_table(all_labels, all_stats, dx, scheme=scheme, t_end=t_end_cached)
        dt_tag = f"_dt{dt_cached:g}".replace(".", "p") if dt_cached is not None else ""
        tag = f"{scheme.lower()}_dx{int(dx)}m{dt_tag}"
        plot_evolution_grid(all_snaps, all_labels, dx,
                            os.path.join(OUT_DIR, f"pc_si_diffcomp_evolution_{tag}.png"),
                            scheme=scheme, t_end=t_end_cached, dt=dt_cached)
        plot_final_snapshot(all_snaps, all_labels, dx,
                            os.path.join(OUT_DIR, f"pc_si_diffcomp_final_{tag}.png"),
                            scheme=scheme, t_end=t_end_cached, dt=dt_cached)
        plot_vertical_profile(all_snaps, all_labels, dx,
                              os.path.join(OUT_DIR, f"pc_si_diffcomp_vprofile_{tag}.png"),
                              scheme=scheme, t_end=t_end_cached, dt=dt_cached)
        return

    dx     = args.dx
    scheme = args.scheme
    kscale = args.kappa_scale
    dt_arg = args.dt   # None -> run_si_variant/_si_kappa fall back to SI_DT
    t_end_arg = args.t_end   # None -> run_si_variant falls back to module T_END
    k2 = _si_kappa(2, dx, dt=dt_arg, scale=kscale)
    k4 = _si_kappa(4, dx, dt=dt_arg, scale=kscale)
    k8 = _si_kappa(8, dx, dt=dt_arg, scale=kscale)
    sp = args.shapiro_period

    print(f"\n{'='*65}")
    print(f"  P&C Exp 1 -- {scheme} diffusion comparison")
    print(f"  dx={dx} m   dt={dt_arg if dt_arg is not None else SI_DT} s   kappa_scale={kscale:.1f}")
    print(f"  T_buoy={T_BUOY:.0f}s  W_scale={W_SCALE:.2f} m/s  3T={3*T_BUOY:.0f}s")
    print(f"  kappa2={k2:.3f} m^2/s   kappa4={k4:.2f} m^4/s   kappa8={k8:.2e} m^8/s")
    print(f"  Shapiro every {sp:.0f} s")
    if scheme in ("SI2", "SI2LU"):
        print(f"  Robert-Asselin filter: alpha=0.1 (applied every step)")
        if scheme == "SI2LU":
            print(f"  Solver: sparse LU factorized once, reused every step "
                  f"(not GMRES) -- same math as SI2, dramatically faster")
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

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    print(f"  Running {len(variants)} variant(s) across {n_workers} "
          f"worker process(es) in parallel...\n")

    results_by_label = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_run_variant_worker,
                            (label, dx, gp, sp, scheme, t_end_arg, dt_arg)): label
            for label, gp in variants
        }
        for fut in as_completed(futures):
            label, snaps, stats = fut.result()
            results_by_label[label] = (snaps, stats)

    all_snaps  = []
    all_stats  = []
    all_labels = []
    for label, gp in variants:
        snaps, stats = results_by_label[label]
        all_snaps.append(snaps)
        all_stats.append(stats)
        all_labels.append(label)

    print_summary_table(all_labels, all_stats, dx, scheme=scheme, t_end=t_end_arg)

    # Cache results BEFORE plotting (a plotting bug should never destroy
    # hours of GMRES compute -- see gr_si_diffusion_comparison.py history).
    import pickle
    kscale_tag = f"_kscale{kscale:.0f}" if kscale != 1.0 else ""
    dt_tag = f"_dt{dt_arg:g}".replace(".", "p") if dt_arg is not None else ""
    tag = f"{scheme.lower()}_dx{int(dx)}m{kscale_tag}{dt_tag}"
    results_dir = os.path.join("output", "results")
    os.makedirs(results_dir, exist_ok=True)
    cache_path = os.path.join(results_dir, f"pc_si_diffcomp_{tag}_cache.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"labels": all_labels, "stats": all_stats,
                    "snaps": all_snaps, "dx": dx, "scheme": scheme,
                    "t_end": t_end_arg, "dt": dt_arg}, f)
    print(f"  Cached results (pre-plotting) -> {cache_path}\n"
          f"  If plotting fails below, regenerate figures with:\n"
          f"    python {os.path.basename(__file__)} --replot-only {cache_path}\n")

    dt_for_titles = dt_arg if dt_arg is not None else SI_DT
    plot_evolution_grid(all_snaps, all_labels, dx,
                        os.path.join(OUT_DIR, f"pc_si_diffcomp_evolution_{tag}.png"),
                        scheme=scheme, t_end=t_end_arg, dt=dt_for_titles)
    plot_final_snapshot(all_snaps, all_labels, dx,
                        os.path.join(OUT_DIR, f"pc_si_diffcomp_final_{tag}.png"),
                        scheme=scheme, t_end=t_end_arg, dt=dt_for_titles)
    plot_vertical_profile(all_snaps, all_labels, dx,
                          os.path.join(OUT_DIR, f"pc_si_diffcomp_vprofile_{tag}.png"),
                          scheme=scheme, t_end=t_end_arg, dt=dt_for_titles)


if __name__ == "__main__":
    main()
