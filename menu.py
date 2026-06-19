"""
menu.py — Interactive scheme runner with save/plot workflow.

Flow:
  1. Select scheme
  2. Enter amplitude, dt, n_steps
  3. Run — shows only last 2 steps
  4. Save prompt → stored as "scheme/amp/nsteps/dt" : "Experiment #N"
  5. Plot prompt → publication-quality contour plots
"""

import numpy as np
import os
import sys
import types
sys.path.insert(0, "src")

# Bypass stale .pyc bytecache (Windows NTFS mount doesn't update mtime on edits)
def _load_src(name, path):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    with open(path) as f:
        source = f.read()
    exec(compile(source, os.path.abspath(path), 'exec'), mod.__dict__)
    return mod

_src = os.path.join(os.path.dirname(__file__), "src")
_load_src("grid",        os.path.join(_src, "grid.py"))
_load_src("dynamics",    os.path.join(_src, "dynamics.py"))
_load_src("integrators", os.path.join(_src, "integrators.py"))
_load_src("results",     os.path.join(_src, "results.py"))

from grid        import Grid
from integrators import step, robert_asselin_filter
from results     import save_experiment, list_experiments

PLOT_DIR    = "output/figures"
RESULTS_DIR = "output/results"
os.makedirs(PLOT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Scheme registry
# ---------------------------------------------------------------------------

SCHEMES = {
    "1": {"name": "FTCS  — Forward Time Centred Space",        "key": "FTCS",
          "order": "1st", "type": "Explicit",
          "stability": "Unconditionally unstable for oscillatory problems."},
    "2": {"name": "BTCS  — Backward Time Centred Space",       "key": "BTCS",
          "order": "1st", "type": "Implicit (1 Picard iter)",
          "stability": "More stable than FTCS. Not fully implicit."},
    "3": {"name": "CTCS  — Leapfrog (Robert-Asselin filter)",  "key": "CTCS",
          "order": "2nd", "type": "Explicit",
          "stability": "Neutral amplitude. Filter applied (alpha=0.1)."},
    "4": {"name": "RK4   — Classical 4-stage Runge-Kutta",     "key": "RK4",
          "order": "4th", "type": "Explicit",
          "stability": "Max stable wDt=2.82. Primary explicit scheme."},
    "5": {"name": "SI    — Semi-Implicit IMEX",                "key": "SI",
          "order": "2nd", "type": "Semi-implicit",
          "stability": "Removes acoustic CFL. L implicit via GMRES."},
    "6": {"name": "EPI2  — Exponential Propagation Iterative", "key": "EPI2",
          "order": "2nd", "type": "Exponential (Krylov)",
          "stability": "e^(L*dt) via Arnoldi. No CFL on L."},
    "7": {"name": "EPI3  — Exponential Propagation Iterative", "key": "EPI3",
          "order": "3rd", "type": "Exponential (Krylov + phi2)",
          "stability": "EPI2 + correction. Pudykiewicz & Clancy 2022."},
}

_experiment_counter = [0]

def _next_experiment_id():
    _experiment_counter[0] += 1
    return _experiment_counter[0]

# ---------------------------------------------------------------------------
# Print menu
# ---------------------------------------------------------------------------

def print_menu():
    print()
    print("=" * 65)
    print("   2D ATMOSPHERIC MODEL — SCHEME SELECTION MENU")
    print("=" * 65)
    print(f"  {'#':<4} {'Scheme':<50} {'Order'}")
    print("-" * 65)
    for num, info in SCHEMES.items():
        print(f"  {num:<4} {info['name']:<50} {info['order']}")
    print("-" * 65)
    print("  s    Show saved experiments")
    print("  0    Exit")
    print("=" * 65)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def _run(scheme_key, bubble_amp, dt, n_steps):
    grid      = Grid({"Lx": 1000.0, "Lz": 1000.0,
                      "dx": 10.0,   "dz": 10.0})
    state     = grid.allocate_state()
    state_old = None  # CTCS bootstraps with FTCS on step 0 when None

    if bubble_amp > 0:
        xc   = grid.Lx / 2.0
        zc   = grid.Lz * 0.4
        r    = 150.0
        r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
        state["theta"] = bubble_amp * np.exp(-r_sq / r**2)

    gamma   = grid.cp / grid.cv
    c_sound = np.sqrt(gamma * grid.Rd * grid.T0)
    cfl     = c_sound * dt / grid.dx
    cfl_tag = "[OK]" if cfl <= 1.0 else "[WARNING: CFL > 1]"

    print(f"\n  Running {scheme_key} | A={bubble_amp}K | "
          f"dt={dt}s | steps={n_steps} | T={dt*n_steps:.1f}s")
    print(f"  CFL = {cfl:.3f}  {cfl_tag}")
    print(f"  {'─'*55}")
    print(f"  (showing last 2 steps only)\n")

    # Save snapshots at 0%, 20%, 40%, 60%, 80%, 100% of run
    save_at  = set(int(n_steps * f) for f in [0.0,0.2,0.4,0.6,0.8,1.0])
    save_at.add(n_steps)
    snapshots = [(0.0, {k: v.copy() for k, v in state.items()})]

    last_two = []
    t        = 0.0
    blown_up = False

    for n in range(n_steps):
        state_prev = state_old   # save phi^(n-1) BEFORE step() overwrites state_old
        try:
            state_new, state_old = step(state, grid, dt,
                                        scheme=scheme_key,
                                        state_old=state_old)
        except Exception as e:
            print(f"  *** ERROR at step {n+1}: {e}")
            blown_up = True
            break

        if scheme_key == 'CTCS' and n > 0:
            # Filter: phi^n_f = phi^n + alpha*(phi^(n-1) - 2*phi^n + phi^(n+1))
            state_filtered = robert_asselin_filter(state_prev, state,
                                                   state_new, alpha=0.1)
            state_old = state_filtered   # filtered phi^n becomes next leapfrog base
            state = state_new
        else:
            state = state_new

        t += dt

        if np.any(~np.isfinite(state['u'])):
            print(f"  *** BLOW-UP at step {n+1}  (t={t:.2f}s) ***")
            print(f"  Amplification factor exceeded 1.")
            blown_up = True
            break

        if (n + 1) in save_at:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

        row = (n+1, t,
               np.max(np.abs(state['u'])),
               np.max(np.abs(state['w'])),
               np.max(np.abs(state['theta'])),
               np.max(np.abs(state['pi'])))
        last_two.append(row)
        if len(last_two) > 2:
            last_two.pop(0)

    # Print last 2 steps
    print(f"  {'step':>6}  {'t(s)':>8}  "
          f"{'|u|_max':>12}  {'|w|_max':>12}  "
          f"{'|th|_max':>12}  {'|pi|_max':>12}")
    print("  " + "─" * 68)
    for row in last_two:
        n_, t_, u_, w_, th_, pi_ = row
        print(f"  {n_:>6}  {t_:>8.2f}  "
              f"{u_:>12.4e}  {w_:>12.4e}  "
              f"{th_:>12.4e}  {pi_:>12.4e}")

    if not blown_up:
        print(f"\n  Run complete. Final t = {t:.2f}s  [No blow-up]")

    return state, snapshots, grid, blown_up

# ---------------------------------------------------------------------------
# Save prompt
# ---------------------------------------------------------------------------

def _prompt_save(scheme_key, bubble_amp, dt, n_steps,
                 state, grid, snapshots):
    print()
    ans = input("  Save this result for future use? (y/n): ").strip().lower()
    if ans != 'y':
        print("  Result not saved.")
        return None

    exp_id   = _next_experiment_id()
    exp_tag  = f"Experiment #{exp_id}"
    dict_key = f"{scheme_key}/{bubble_amp}/{n_steps}/{dt}"
    file_name = (f"{scheme_key}_A{bubble_amp}_n{n_steps}_dt{dt}"
                 ).replace(".", "p")

    save_experiment(
        file_name, state, grid,
        snapshots = snapshots,
        metadata  = {
            "experiment_id":  exp_id,
            "experiment_tag": exp_tag,
            "dict_key":       dict_key,
            "scheme":         scheme_key,
            "bubble_amp":     bubble_amp,
            "dt":             dt,
            "n_steps":        n_steps,
            "t_final":        dt * n_steps,
        }
    )
    print(f"\n  Saved  →  key : \"{dict_key}\"")
    print(f"             tag : \"{exp_tag}\"")
    print(f"             file: {file_name}.npz")
    return dict_key

# ---------------------------------------------------------------------------
# Publication-quality contour plot
# ---------------------------------------------------------------------------

def _contour_plot(snapshots, grid, variable, scheme_key,
                  bubble_amp, dt):
    """
    Produce filled contour + contour line plot.
    theta' — Robert (1993) style (red filled, black lines)
    w       — velocity plot (RdBu diverging, pos/neg lines)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0
    X, Z = np.meshgrid(x_km, z_km)

    n_snaps   = len(snapshots)
    n_cols    = min(n_snaps, 3)
    n_rows    = (n_snaps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.0*n_cols, 4.5*n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    # Colour limits consistent across all panels
    all_vals = np.concatenate(
        [s[variable].ravel() for _, s in snapshots]
    )
    vmax = max(np.max(np.abs(all_vals)), 1e-10)

    if variable == 'theta':
        cmap         = 'RdYlBu_r'
        levels_fill  = np.linspace(0, vmax, 20)
        levels_line  = np.linspace(0.1*vmax, vmax, 7)
        cbar_label   = "θ'  (K)"
        line_color   = 'black'
        vmin         = 0.0
    else:
        cmap         = 'RdBu_r'
        levels_fill  = np.linspace(-vmax, vmax, 21)
        levels_line  = np.linspace(0.1*vmax, vmax, 6)
        cbar_label   = "w  (m/s)"
        line_color   = 'darkred'
        vmin         = -vmax

    for idx, (t, state) in enumerate(snapshots):
        row = idx // n_cols
        col = idx  % n_cols
        ax  = axes[row][col]
        ax.set_facecolor('#f5f5f5')

        data = state[variable]

        # Filled contours
        cf = ax.contourf(X, Z, data,
                         levels=levels_fill,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         alpha=0.88)

        # Contour lines + labels
        cl = ax.contour(X, Z, data,
                        levels=levels_line,
                        colors=line_color,
                        linewidths=0.7, alpha=0.75)
        ax.clabel(cl, fmt='%.2g', fontsize=6,
                  inline=True, inline_spacing=2)

        # For w: also draw negative dashed lines
        if variable == 'w':
            levels_neg = np.linspace(-vmax, -0.1*vmax, 6)
            ax.contour(X, Z, data,
                       levels=levels_neg,
                       colors='navy',
                       linewidths=0.7,
                       linestyles='dashed', alpha=0.75)

        # Sponge layer marker
        ax.axhline(0.80, color='steelblue', lw=0.9,
                   linestyle='--', alpha=0.5)
        ax.text(0.02, 0.81, 'sponge', transform=ax.transAxes,
                fontsize=6, color='steelblue', alpha=0.7)

        t_label = f"t = {t/60:.1f} min" if t >= 60 else f"t = {t:.2f} s"
        ax.set_title(t_label, fontsize=11,
                     fontweight='bold', pad=4)
        ax.set_xlabel("x  (km)", fontsize=9)
        ax.set_ylabel("z  (km)", fontsize=9)
        ax.set_xlim(0, grid.Lx/1000)
        ax.set_ylim(0, grid.Lz/1000)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.18, lw=0.4)

        cbar = fig.colorbar(cf, ax=ax,
                            fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_snaps, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    var_label = "θ'" if variable == 'theta' else "w"
    title_extra = ("" if variable == 'theta'
                   else "  (red=upward, blue=downward)")
    fig.suptitle(
        f"Warm Bubble — {var_label}{title_extra}\n"
        f"Scheme: {scheme_key}  |  A = {bubble_amp} K  |  Δt = {dt} s",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    safe = (f"{scheme_key}_A{str(bubble_amp).replace('.','p')}"
            f"_{variable}")
    fname = os.path.join(PLOT_DIR, f"{safe}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor='white')
    print(f"  Plot saved → {fname}")
    plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Raw grid (pcolormesh) plot — diagnostic style
# ---------------------------------------------------------------------------

def _pcolormesh_plot(snapshots, grid, variable, scheme_key,
                     bubble_amp, dt):
    """
    Raw grid-cell heatmap using pcolormesh.

    Use case: diagnostics / zero-amplitude tests / spotting
    grid-scale artifacts (NaNs, single-cell noise, checkerboarding)
    that a smoothed contour plot would hide.
    """
    import matplotlib.pyplot as plt

    x_km = grid.x_1d / 1000.0
    z_km = grid.z_1d / 1000.0

    n_snaps   = len(snapshots)
    n_cols    = min(n_snaps, 3)
    n_rows    = (n_snaps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2*n_cols, 4.0*n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    all_vals = np.concatenate(
        [s[variable].ravel() for _, s in snapshots]
    )
    vmax = max(np.max(np.abs(all_vals)), 1e-12)

    if variable == 'theta':
        cmap, vmin, cbar_label = 'hot_r',  0.0,  "θ'  (K)"
    else:
        cmap, vmin, cbar_label = 'RdBu_r', -vmax, "w  (m/s)"

    for idx, (t, state) in enumerate(snapshots):
        row, col = idx // n_cols, idx % n_cols
        ax  = axes[row][col]
        data = state[variable]

        im = ax.pcolormesh(x_km, z_km, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto')

        ax.axhline(0.80, color='dodgerblue', lw=0.8,
                   linestyle='--', alpha=0.6)
        t_label = f"t = {t/60:.1f} min" if t >= 60 else f"t = {t:.2f} s"
        ax.set_title(t_label, fontsize=10, fontweight='bold')
        ax.set_xlabel("x  (km)", fontsize=8)
        ax.set_ylabel("z  (km)", fontsize=8)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    for idx in range(n_snaps, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    var_label = "θ'" if variable == 'theta' else "w"
    fig.suptitle(
        f"Raw Grid (pcolormesh) — {var_label}\n"
        f"Scheme: {scheme_key}  |  A = {bubble_amp} K  |  Δt = {dt} s",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    safe  = (f"{scheme_key}_A{str(bubble_amp).replace('.','p')}"
             f"_{variable}_raw")
    fname = os.path.join(PLOT_DIR, f"{safe}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Plot saved → {fname}")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Plot prompt
# ---------------------------------------------------------------------------

def _prompt_plot(scheme_key, bubble_amp, dt,
                 snapshots, grid, blown_up):
    if blown_up:
        print("  Run ended in blow-up — no plots generated.")
        return

    print()
    print("  Generate plots?")
    print("  Contour (publication style, Robert 1993 / Clancy 2022):")
    print("    1  θ' contour")
    print("    2  w contour")
    print("    3  Both contours")
    print("  Raw grid (pcolormesh, diagnostic — shows grid-cell detail):")
    print("    4  θ' pcolormesh")
    print("    5  w pcolormesh")
    print("    6  Both pcolormesh")
    print("    7  Everything (all 4 plots)")
    print("  0  Skip")

    choice = input("  Choice: ").strip()
    if choice == "0":
        return

    if choice in ("1", "3", "7"):
        _contour_plot(snapshots, grid, 'theta',
                      scheme_key, bubble_amp, dt)
    if choice in ("2", "3", "7"):
        _contour_plot(snapshots, grid, 'w',
                      scheme_key, bubble_amp, dt)
    if choice in ("4", "6", "7"):
        _pcolormesh_plot(snapshots, grid, 'theta',
                        scheme_key, bubble_amp, dt)
    if choice in ("5", "6", "7"):
        _pcolormesh_plot(snapshots, grid, 'w',
                        scheme_key, bubble_amp, dt)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("\n  Welcome to the 2D Atmospheric Model.")

    while True:
        print_menu()
        choice = input("  Enter choice: ").strip().lower()

        if choice == "0":
            print("\n  Exiting.\n")
            break

        if choice == "s":
            list_experiments()
            input("  Press Enter to continue...")
            continue

        if choice not in SCHEMES:
            print(f"  '{choice}' not valid.")
            continue

        info = SCHEMES[choice]
        print(f"\n  Selected : {info['name']}")
        print(f"  Type     : {info['type']}")
        print(f"  Order    : {info['order']}")
        print(f"  Stability: {info['stability']}")

        print("\n  Run parameters (press Enter for default):")
        def ask(prompt, default, cast):
            v = input(f"    {prompt} [{default}]: ").strip()
            return cast(v) if v else default

        bubble_amp = ask("Bubble amplitude in K  (0 = zero test)", 2.0, float)
        dt         = ask("Time step dt  [s]",                      0.01, float)
        n_steps    = ask("Number of steps (T=dt*n, e.g. 5000→50s @ dt=0.01)", 5000, int)

        if dt <= 0:
            print("  dt must be > 0.")
            continue

        state, snapshots, grid, blown = _run(
            info["key"], bubble_amp, dt, n_steps
        )

        if blown:
            input("\n  Press Enter to return to menu...")
            continue

        _prompt_save(info["key"], bubble_amp, dt, n_steps,
                     state, grid, snapshots)

        _prompt_plot(info["key"], bubble_amp, dt,
                     snapshots, grid, blown)

        input("\n  Press Enter to return to menu...")


if __name__ == "__main__":
    main()
