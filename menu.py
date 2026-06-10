"""
menu.py
=======
Interactive scheme selection menu.
To add a new scheme: add entry to SCHEMES dict, implement in integrators.py.
"""

import numpy as np
import sys
sys.path.insert(0, "src")

from grid        import Grid
from dynamics    import compute_rhs
from integrators import step, robert_asselin_filter


SCHEMES = {
    "1": {
        "name":      "FTCS  — Forward Time Centred Space",
        "key":       "FTCS",
        "order":     "1st order",
        "type":      "Explicit",
        "stability": "Unstable for oscillatory problems. Expect blow-up.",
        "status":    "ready",
    },
    "2": {
        "name":      "BTCS  — Backward Time Centred Space",
        "key":       "BTCS",
        "order":     "1st order",
        "type":      "Implicit (linearised, 1 Picard iter)",
        "stability": "More stable than FTCS. Not fully implicit.",
        "status":    "ready",
    },
    "3": {
        "name":      "CTCS  — Centred Time Centred Space (Leapfrog)",
        "key":       "CTCS",
        "order":     "2nd order",
        "type":      "Explicit",
        "stability": "Neutral amplitude. Robert-Asselin filter applied.",
        "status":    "ready",
    },
    "4": {
        "name":      "RK4   — Classical 4-stage Runge-Kutta",
        "key":       "RK4",
        "order":     "4th order",
        "type":      "Explicit",
        "stability": "Most stable explicit scheme. Max wDt ~ 2.82.",
        "status":    "ready",
    },
    "5": {
        "name":      "SI    — Semi-Implicit IMEX (Crank-Nicolson)",
        "key":       "SI",
        "order":     "2nd order",
        "type":      "Semi-implicit (L implicit, N explicit)",
        "stability": "Removes acoustic CFL. dt limited by advective speed.",
        "status":    "ready",
    },
    "6": {
        "name":      "EPI2  — Exponential Propagation Iterative",
        "key":       "EPI2",
        "order":     "2nd order",
        "type":      "Exponential (Krylov subspace)",
        "stability": "Exact e^(L*dt) via Arnoldi. No CFL on L.",
        "status":    "ready",
    },
    "7": {
        "name":      "EPI3  — Exponential Propagation Iterative",
        "key":       "EPI3",
        "order":     "3rd order",
        "type":      "Exponential (Krylov + correction)",
        "stability": "EPI2 + phi2 correction. Pudykiewicz & Clancy 2022.",
        "status":    "ready",
    },
}


def print_menu():
    print()
    print("=" * 65)
    print("   2D ATMOSPHERIC MODEL — SCHEME SELECTION MENU")
    print("=" * 65)
    print(f"  {'#':<4} {'Scheme':<50} {'Status'}")
    print("-" * 65)
    for num, info in SCHEMES.items():
        print(f"  {num:<4} {info['name']:<50} {info['status']}")
    print("-" * 65)
    print("  0    Exit")
    print("=" * 65)


def print_scheme_info(info):
    print(f"\n  Selected : {info['name']}")
    print(f"  Type     : {info['type']}")
    print(f"  Order    : {info['order']}")
    print(f"  Stability: {info['stability']}\n")


def get_run_params():
    def ask(prompt, default, cast):
        val = input(f"    {prompt} [{default}]: ").strip()
        return cast(val) if val else default

    print("  Run parameters (press Enter to use default):")
    bubble_amp = ask("Warm bubble amplitude in K (0 = zero test)", 0.0, float)
    dt         = ask("Time step dt [s]", 0.02, float)
    n_steps    = ask("Number of steps", 200, int)
    return bubble_amp, dt, n_steps


def run_model(scheme_key, bubble_amp, dt, n_steps):
    print(f"\n  scheme={scheme_key}  A={bubble_amp}K  "
          f"dt={dt}s  steps={n_steps}  T={dt*n_steps:.1f}s")

    grid      = Grid({"Lx": 1000.0, "Lz": 1000.0, "dx": 10.0, "dz": 10.0})
    state     = grid.allocate_state()
    state_old = grid.allocate_state()

    if bubble_amp > 0:
        xc   = grid.Lx / 2.0
        zc   = grid.Lz * 0.4
        r    = 150.0
        r_sq = (grid.x_2d - xc)**2 + (grid.z_2d - zc)**2
        state["theta"] = bubble_amp * np.exp(-r_sq / r**2)
        print(f"  Warm bubble: centre=({xc}m, {zc}m) r={r}m")
    else:
        print("  Zero-amplitude test — state should stay at zero.")

    gamma   = grid.cp / grid.cv
    c_sound = np.sqrt(gamma * grid.Rd * grid.T0)
    cfl     = c_sound * dt / grid.dx
    print(f"  CFL = {cfl:.3f}  (c_sound={c_sound:.1f} m/s)",
          "[OK]" if cfl <= 1.0 else "[WARNING: CFL > 1]")
    print()

    blown_up = False
    for n in range(n_steps):
        try:
            state_new, state_old = step(state, grid, dt,
                                        scheme=scheme_key,
                                        state_old=state_old)
        except Exception as e:
            print(f"  *** ERROR at step {n+1}: {e}")
            blown_up = True
            break

        if scheme_key == 'CTCS' and n > 0:
            state = robert_asselin_filter(state_old, state,
                                          state_new, alpha=0.1)
        else:
            state = state_new

        for key, val in state.items():
            if np.any(np.isnan(val)) or np.any(np.isinf(val)):
                print(f"  *** BLOW-UP at step {n+1} "
                      f"(t={(n+1)*dt:.1f}s): {key} is NaN/Inf ***")
                blown_up = True
                break
        if blown_up:
            break

        if (n + 1) % max(1, n_steps // 10) == 0:
            u_max  = np.max(np.abs(state['u']))
            w_max  = np.max(np.abs(state['w']))
            th_max = np.max(np.abs(state['theta']))
            pi_max = np.max(np.abs(state['pi']))
            print(f"  step {n+1:5d}  t={(n+1)*dt:7.2f}s  "
                  f"|u|={u_max:.3e}  |w|={w_max:.3e}  "
                  f"|θ'|={th_max:.3e}  |π'|={pi_max:.3e}")

    if not blown_up:
        print(f"\n  Run complete. t={n_steps*dt:.1f}s  [No blow-up]")


def main():
    print("\n  Welcome to the 2D Atmospheric Model.")

    while True:
        print_menu()
        choice = input("  Enter scheme number (0 to exit): ").strip()

        if choice == "0":
            print("\n  Exiting.\n")
            break

        if choice not in SCHEMES:
            print(f"\n  '{choice}' is not valid. Try again.")
            continue

        info = SCHEMES[choice]
        print_scheme_info(info)

        bubble_amp, dt, n_steps = get_run_params()
        run_model(info["key"], bubble_amp, dt, n_steps)

        input("\n  Press Enter to return to menu...")


if __name__ == "__main__":
    main()
