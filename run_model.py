"""
run_model.py
============
Command-line runner with result saving and regulated output.

Changes from previous version:
  - Only prints final summary by default (--verbose for step-by-step)
  - Saves results to output/results/ using results.py
  - Accepts --print_every to control output frequency

Usage
-----
    # Quiet — just print final summary
    python run_model.py --scheme CTCS --bubble_amp 2.0 --dt 0.02 --n_steps 500

    # Verbose — print every 50 steps
    python run_model.py --scheme CTCS --bubble_amp 2.0 --dt 0.02 --n_steps 500
                        --verbose --print_every 50

    # Save results
    python run_model.py --scheme CTCS --bubble_amp 2.0 --dt 0.02 --n_steps 500
                        --save --name ctcs_bubble2_dt002
"""

import numpy as np
import argparse
import sys
sys.path.insert(0, "src")

from grid        import Grid
from integrators import step, robert_asselin_filter
from results     import save_experiment


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def set_initial_conditions(state, grid, bubble_amp=2.0,
                           bubble_r=150.0):
    if bubble_amp > 0:
        xc   = grid.Lx / 2.0
        zc   = grid.Lz * 0.4
        r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
        state["theta"] = bubble_amp * np.exp(-r_sq / bubble_r**2)
    return state


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(scheme='CTCS', bubble_amp=2.0, dt=0.02, n_steps=500,
        verbose=False, print_every=50, save=False, name=None):

    print(f"\n{'='*55}")
    print(f"  Scheme: {scheme}  |  A={bubble_amp}K  "
          f"dt={dt}s  steps={n_steps}  T={dt*n_steps:.1f}s")
    print(f"{'='*55}")

    grid      = Grid({"Lx": 1000.0, "Lz": 1000.0,
                      "dx": 10.0,   "dz": 10.0})
    state     = grid.allocate_state()
    state_old = None  # CTCS bootstraps with FTCS on step 0 when None

    state = set_initial_conditions(state, grid, bubble_amp)

    # CFL check
    gamma   = grid.cp / grid.cv
    c_sound = np.sqrt(gamma * grid.Rd * grid.T0)
    cfl     = c_sound * dt / grid.dx
    cfl_str = "[OK]" if cfl <= 1.0 else "[WARNING: CFL > 1]"
    print(f"  CFL = {cfl:.3f}  {cfl_str}")

    if bubble_amp == 0:
        print("  Zero-amplitude test: state should remain at zero.")

    # Snapshot storage for saving
    snapshots = [(0.0, {k: v.copy() for k, v in state.items()})]

    # Time loop
    t        = 0.0
    blown_up = False

    for n in range(n_steps):
        state_new, state_old = step(state, grid, dt,
                                    scheme=scheme,
                                    state_old=state_old)

        if scheme == 'CTCS' and n > 0:
            state = robert_asselin_filter(state_old, state,
                                          state_new, alpha=0.1)
        else:
            state = state_new

        t += dt

        # Blow-up check every step
        if np.any(np.isnan(state['u'])) or np.any(np.isinf(state['u'])):
            print(f"\n  *** BLOW-UP at step {n+1} (t={t:.2f}s) ***")
            print(f"  Amplification factor exceeded 1.")
            blown_up = True
            break

        # Verbose step output
        if verbose and (n + 1) % print_every == 0:
            u_max = np.max(np.abs(state['u']))
            w_max = np.max(np.abs(state['w']))
            th_max = np.max(np.abs(state['theta']))
            pi_max = np.max(np.abs(state['pi']))
            print(f"  step {n+1:5d}  t={t:7.2f}s  "
                  f"|u|={u_max:.3e}  |w|={w_max:.3e}  "
                  f"|θ'|={th_max:.3e}  |π'|={pi_max:.3e}")

        # Save snapshot every 10% of run
        if (n + 1) % max(1, n_steps // 10) == 0:
            snapshots.append((t, {k: v.copy() for k, v in state.items()}))

    # Final summary — always printed
    print(f"\n  {'─'*45}")
    if not blown_up:
        u_max  = np.max(np.abs(state['u']))
        w_max  = np.max(np.abs(state['w']))
        th_max = np.max(np.abs(state['theta']))
        pi_max = np.max(np.abs(state['pi']))
        print(f"  Final t = {t:.2f}s  [No blow-up]")
        print(f"  |u|_max  = {u_max:.4e}  m/s")
        print(f"  |w|_max  = {w_max:.4e}  m/s")
        print(f"  |θ'|_max = {th_max:.4e}  K")
        print(f"  |π'|_max = {pi_max:.4e}")
    else:
        print(f"  Run stopped at t={t:.2f}s due to blow-up.")
    print(f"  {'─'*45}")

    # Save results
    if save and not blown_up:
        exp_name = name or f"{scheme}_A{bubble_amp}_dt{dt}_n{n_steps}"
        save_experiment(
            name      = exp_name,
            state     = state,
            grid      = grid,
            snapshots = snapshots,
            metadata  = {
                "scheme":     scheme,
                "bubble_amp": bubble_amp,
                "dt":         dt,
                "n_steps":    n_steps,
                "t_final":    t,
                "cfl":        cfl,
                "blown_up":   blown_up,
            }
        )

    return state, grid, snapshots


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--scheme",      default="CTCS",
                   choices=["FTCS","BTCS","CTCS","RK4"])
    p.add_argument("--bubble_amp",  type=float, default=2.0)
    p.add_argument("--dt",          type=float, default=0.02)
    p.add_argument("--n_steps",     type=int,   default=500)
    p.add_argument("--verbose",     action="store_true",
                   help="Print diagnostics every --print_every steps")
    p.add_argument("--print_every", type=int,   default=50,
                   help="Steps between diagnostic prints (verbose mode)")
    p.add_argument("--save",        action="store_true",
                   help="Save results to output/results/")
    p.add_argument("--name",        default=None,
                   help="Name for saved experiment")
    args = p.parse_args()

    run(scheme      = args.scheme,
        bubble_amp  = args.bubble_amp,
        dt          = args.dt,
        n_steps     = args.n_steps,
        verbose     = args.verbose,
        print_every = args.print_every,
        save        = args.save,
        name        = args.name)
