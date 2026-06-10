"""
dynamics.py
===========
Computes the RHS of the governing equations (Giraldo & Restelli 2008).

Now exposes THREE functions:
  compute_rhs(state, grid)          — full RHS = L*q + N(q)
  compute_linear_rhs(state, grid)   — linear part L*q only
  compute_nonlinear_rhs(state, grid)— nonlinear part N(q) only

This split is needed for semi-implicit and EPI schemes:
  - Semi-implicit: treats L implicitly, N explicitly
  - EPI2/EPI3: computes e^(L*dt) exactly via Krylov, N explicitly

Linear terms L (stiff — acoustic waves, buoyancy, base-state gradient):
  u  eq:  -cp * theta_bar * dpi'/dx
  w  eq:  -cp * theta_bar * dpi'/dz  +  g * theta' / theta_bar
  pi eq:  -(R/cv) * pi_bar * (du/dx + dw/dz)  +  g*w / (cp*theta_bar)
  th eq:  -w * dtheta_bar/dz

Nonlinear terms N(q) (advection — slower, handled explicitly):
  u  eq:  -u*du/dx - w*du/dz - cp*theta'*dpi'/dx
  w  eq:  -u*dw/dx - w*dw/dz - cp*theta'*dpi'/dz
  pi eq:  -u*dpi'/dx - w*dpi'/dz - (R/cv)*pi'*(du/dx + dw/dz)
  th eq:  -u*dtheta'/dx - w*dtheta'/dz
"""

import numpy as np


def compute_rhs(state, grid):
    """Full RHS = L*q + N(q). Used by FTCS, BTCS, CTCS, RK4."""
    lin = compute_linear_rhs(state, grid)
    nln = compute_nonlinear_rhs(state, grid)
    return {k: lin[k] + nln[k] for k in lin}


def compute_linear_rhs(state, grid):
    """
    Linear part L*q — contains stiff acoustic and buoyancy terms.
    Used directly by semi-implicit and EPI schemes.
    """
    u     = state["u"]
    w     = state["w"]
    theta = state["theta"]
    pi    = state["pi"]

    dx = grid.dx
    dz = grid.dz
    cp = grid.cp
    cv = grid.cv
    g  = grid.g

    theta_bar     = grid.theta_bar[:, np.newaxis]
    pi_bar        = grid.pi_bar[:, np.newaxis]
    dtheta_bar_dz = grid.dtheta_bar_dz[:, np.newaxis]

    # Spatial derivatives of perturbations
    dpi_dx  = _dx(pi,    dx)
    dpi_dz  = _dz(pi,    dz)
    du_dx   = _dx(u,     dx)
    dw_dz   = _dz(w,     dz)

    # Linear acoustic pressure gradient in u
    rhs_u_lin = - cp * theta_bar * dpi_dx

    # Linear pressure gradient + buoyancy in w
    rhs_w_lin = (- cp * theta_bar * dpi_dz
                 + g * theta / theta_bar)

    # Linear acoustic source (divergence) in pi
    divergence  = du_dx + dw_dz
    rhs_pi_lin  = (- (grid.Rd / cv) * pi_bar * divergence
                   + g * w / (cp * theta_bar))

    # Linear base-state advection in theta
    rhs_theta_lin = - w * dtheta_bar_dz

    # Apply boundary conditions
    rhs_w_lin[0,  :] = 0.0
    rhs_w_lin[-1, :] = 0.0
    rhs_theta_lin[0,  :] = 0.0
    rhs_theta_lin[-1, :] = 0.0

    return {
        "u":     rhs_u_lin,
        "w":     rhs_w_lin,
        "theta": rhs_theta_lin,
        "pi":    rhs_pi_lin,
    }


def compute_nonlinear_rhs(state, grid):
    """
    Nonlinear part N(q) — contains advection terms.
    These are slower than acoustic waves and treated explicitly.
    """
    u     = state["u"]
    w     = state["w"]
    theta = state["theta"]
    pi    = state["pi"]

    dx = grid.dx
    dz = grid.dz
    cp = grid.cp
    cv = grid.cv

    theta_bar = grid.theta_bar[:, np.newaxis]
    pi_bar    = grid.pi_bar[:, np.newaxis]

    # All spatial derivatives
    du_dx    = _dx(u,     dx)
    du_dz    = _dz(u,     dz)
    dw_dx    = _dx(w,     dx)
    dw_dz    = _dz(w,     dz)
    dtheta_dx= _dx(theta, dx)
    dtheta_dz= _dz(theta, dz)
    dpi_dx   = _dx(pi,    dx)
    dpi_dz   = _dz(pi,    dz)

    # Nonlinear advection of u
    rhs_u_nln = - u * du_dx - w * du_dz - cp * theta * dpi_dx

    # Nonlinear advection of w
    rhs_w_nln = - u * dw_dx - w * dw_dz - cp * theta * dpi_dz

    # Nonlinear advection of pi + nonlinear acoustic
    divergence   = du_dx + dw_dz
    rhs_pi_nln   = (- u * dpi_dx
                    - w * dpi_dz
                    - (grid.Rd / cv) * pi * divergence)

    # Nonlinear advection of theta
    rhs_theta_nln = - u * dtheta_dx - w * dtheta_dz

    # Boundary conditions
    rhs_w_nln[0,  :] = 0.0
    rhs_w_nln[-1, :] = 0.0
    rhs_theta_nln[0,  :] = 0.0
    rhs_theta_nln[-1, :] = 0.0

    return {
        "u":     rhs_u_nln,
        "w":     rhs_w_nln,
        "theta": rhs_theta_nln,
        "pi":    rhs_pi_nln,
    }


# ---------------------------------------------------------------------------
# Finite difference helpers
# ---------------------------------------------------------------------------

def _dx(f, dx):
    """2nd-order centred x-derivative with periodic BCs via np.roll."""
    return (np.roll(f, -1, axis=1) - np.roll(f, +1, axis=1)) / (2 * dx)


def _dz(f, dz):
    """
    z-derivative: 2nd-order centred interior,
    1st-order one-sided at top/bottom boundaries.
    """
    dfdz = np.zeros_like(f)
    dfdz[1:-1, :] = (f[2:, :]  - f[:-2, :]) / (2 * dz)
    dfdz[0,    :] = (f[1,  :]  - f[0,   :]) / dz
    dfdz[-1,   :] = (f[-1, :]  - f[-2,  :]) / dz
    return dfdz


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from grid import Grid

    g     = Grid()
    state = g.allocate_state()

    rhs     = compute_rhs(state, g)
    rhs_lin = compute_linear_rhs(state, g)
    rhs_nln = compute_nonlinear_rhs(state, g)

    print("Zero amplitude test:")
    for key in rhs:
        full  = np.max(np.abs(rhs[key]))
        lin   = np.max(np.abs(rhs_lin[key]))
        nln   = np.max(np.abs(rhs_nln[key]))
        print(f"  {key:6s}  full={full:.1e}  linear={lin:.1e}  nonlinear={nln:.1e}")
