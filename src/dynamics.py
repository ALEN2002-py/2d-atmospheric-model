"""
dynamics.py
===========
Computes the RHS of the governing equations (Giraldo & Restelli 2008).

Exposes three functions:
  compute_rhs(state, grid)           — full RHS = L*q + N(q)
  compute_linear_rhs(state, grid)    — linear stiff part L*q
  compute_nonlinear_rhs(state, grid) — nonlinear advection N(q)

Finite differences
------------------
  _dx(f, dx)  — 2nd-order centred x-derivative, periodic BCs
  _dz(f, dz)  — 2nd-order centred z-derivative, 1st-order at boundaries

Numba JIT
---------
  If Numba is installed, _dx_nb and _dz_nb replace _dx and _dz.
  These are compiled to machine code on first call — subsequent calls
  are 10-50x faster than the pure numpy versions on large grids.

  Install:  pip install numba
  The code detects Numba automatically — no changes needed.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Numba JIT — compiled finite difference kernels
# ---------------------------------------------------------------------------

try:
    from numba import njit

    @njit(cache=True)
    def _dx_nb(f, dx):
        """
        2nd-order centred x-derivative with periodic BCs.
        Numba-compiled — avoids creating temporary arrays like np.roll does.
        """
        nz, nx = f.shape
        out    = np.empty((nz, nx))
        inv2dx = 1.0 / (2.0 * dx)

        for k in range(nz):
            for i in range(nx):
                ip1      = (i + 1) % nx   # wrap right
                im1      = (i - 1) % nx   # wrap left
                out[k,i] = (f[k, ip1] - f[k, im1]) * inv2dx
        return out

    @njit(cache=True)
    def _dz_nb(f, dz):
        """
        z-derivative: 2nd-order centred interior,
        1st-order one-sided at top/bottom boundaries.
        Numba-compiled.
        """
        nz, nx  = f.shape
        out     = np.empty((nz, nx))
        inv2dz  = 1.0 / (2.0 * dz)
        invdz   = 1.0 / dz

        # Interior: centred
        for k in range(1, nz - 1):
            for i in range(nx):
                out[k, i] = (f[k+1, i] - f[k-1, i]) * inv2dz

        # Bottom boundary (k=0): forward difference
        for i in range(nx):
            out[0, i] = (f[1, i] - f[0, i]) * invdz

        # Top boundary (k=nz-1): backward difference
        for i in range(nx):
            out[nz-1, i] = (f[nz-1, i] - f[nz-2, i]) * invdz

        return out

    # Assign compiled versions as the active functions
    _dx = _dx_nb
    _dz = _dz_nb
    NUMBA_ACTIVE = True
    print("  [dynamics] Numba JIT enabled for _dx and _dz")

except ImportError:
    # Fall back to pure numpy — correct but slower
    NUMBA_ACTIVE = False

    def _dx(f, dx):
        """2nd-order centred x-derivative, periodic via np.roll."""
        return (np.roll(f, -1, axis=1) - np.roll(f, +1, axis=1)) / (2*dx)

    def _dz(f, dz):
        """z-derivative: centred interior, one-sided boundaries."""
        out         = np.empty_like(f)
        out[1:-1,:] = (f[2:,:]  - f[:-2,:]) / (2*dz)
        out[0,   :] = (f[1, :]  - f[0,  :]) / dz
        out[-1,  :] = (f[-1,:]  - f[-2, :]) / dz
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rhs(state, grid):
    """
    Full RHS = L*q + N(q) [+ hyperdiffusion if configured].
    Used by FTCS, BTCS, CTCS, RK4.

    Diffusion is controlled by two grid attributes:
      grid.diffusion_coeff  — κ coefficient [m^order / s], default 0
      grid.diffusion_order  — 2, 4, or 8,                  default 2
    If diffusion_coeff == 0, no diffusion is added regardless of order.
    """
    lin = compute_linear_rhs(state, grid)
    nln = compute_nonlinear_rhs(state, grid)
    rhs = {k: lin[k] + nln[k] for k in lin}

    kappa = getattr(grid, 'diffusion_coeff', 0.0)
    if kappa > 0.0:
        order = getattr(grid, 'diffusion_order', 2)
        diff  = compute_hyperdiffusion_rhs(state, grid, order=order, coeff=kappa)
        for k in rhs:
            rhs[k] += diff[k]

    return rhs


def _laplacian(f, dx, dz):
    """
    2nd-order finite-difference Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂z².

    Boundary conditions:
      x: periodic
      z: zero-Neumann (ghost = edge value, i.e. no-flux)

    This is a helper used by compute_hyperdiffusion_rhs to build
    higher-order operators by repeated application.
    """
    # ∂²f/∂x² — periodic
    d2x = (np.roll(f, -1, axis=1) - 2.0*f + np.roll(f, 1, axis=1)) / dx**2

    # ∂²f/∂z² — zero-Neumann at top/bottom
    d2z          = np.empty_like(f)
    d2z[1:-1, :] = (f[2:, :]  - 2.0*f[1:-1, :] + f[:-2, :]) / dz**2
    d2z[0,    :] = (f[1,  :]  - f[0,  :])                    / dz**2
    d2z[-1,   :] = (f[-2, :]  - f[-1, :])                    / dz**2

    return d2x + d2z


def compute_hyperdiffusion_rhs(state, grid, order=2, coeff=None):
    """
    Hyperdiffusion tendency:  (-1)^(n+1) * kappa * nabla^(2n) q,  n = order/2.

    Sign convention keeps the operator dissipative for all orders:
      order=2 ->  +kappa * nabla^2   (standard Laplacian)
      order=4 ->  -kappa * nabla^4   (biharmonic)
      order=8 ->  -kappa * nabla^8   (octaharmonic)

    In spectral space, nabla^(2n) has eigenvalue (-1)^n * k^(2n), so we need
    sign = (-1)^(n+1) to guarantee a negative tendency eigenvalue (damping).

    Selectivity: the ratio of damping at k=pi/dx vs k=pi/(2dx) is 4^n,
    so order=8 damps grid-scale noise 256x more than waves at twice the
    grid spacing, leaving the bubble structure essentially untouched.

    order must be 2, 4, or 8. coeff defaults to grid.diffusion_coeff.
    """
    if coeff is None:
        coeff = getattr(grid, 'diffusion_coeff', 0.0)

    if order not in (2, 4, 8):
        raise ValueError(f"Hyperdiffusion order must be 2, 4, or 8; got {order}")

    dx = grid.dx
    dz = grid.dz
    n  = order // 2                         # number of times to apply ∇²
    sign = (-1) ** (n + 1)                  # +1 for order=2, -1 for order=4,8

    out = {}
    for key in ('u', 'w', 'theta', 'pi'):
        f = state[key]
        # Apply ∇² repeatedly n times
        result = f
        for _ in range(n):
            result = _laplacian(result, dx, dz)
        out[key] = sign * coeff * result

    # Solid-wall BC: zero diffusive flux of w at top/bottom
    out['w'][0,  :] = 0.0
    out['w'][-1, :] = 0.0

    return out


def compute_diffusion_rhs(state, grid):
    """
    Explicit diffusion  κ ∇²q  for all prognostic variables.
    κ = grid.diffusion_coeff  [m²/s]

    G&R (2008) use κ ≈ 75 m²/s to smooth grid-scale noise.
    Stability limit (explicit):  dt ≤ dx² / (4κ)
      → at dx=10 m, κ=75:  dt_max ≈ 0.33 s  (RK4 at dt=0.01 s is well within this)

    Finite-difference Laplacian:
      x: periodic  (np.roll)
      z: zero-Neumann (no-flux) at top and bottom
    """
    kappa = grid.diffusion_coeff
    dx    = grid.dx
    dz    = grid.dz
    out   = {}

    for key in ('u', 'w', 'theta', 'pi'):
        f = state[key]

        # ∂²f/∂x² — periodic BCs
        d2x = (np.roll(f, -1, axis=1) - 2.0*f + np.roll(f, 1, axis=1)) / dx**2

        # ∂²f/∂z² — zero-Neumann BCs (ghost point = boundary value)
        d2z          = np.empty_like(f)
        d2z[1:-1, :] = (f[2:, :]  - 2.0*f[1:-1, :] + f[:-2, :]) / dz**2
        d2z[0,    :] = (f[1,  :]  - f[0,  :])                    / dz**2
        d2z[-1,   :] = (f[-2, :]  - f[-1, :])                    / dz**2

        out[key] = kappa * (d2x + d2z)

    # Enforce solid-wall BC: no diffusive flux of w at top/bottom
    out['w'][0,  :] = 0.0
    out['w'][-1, :] = 0.0

    return out


def compute_linear_rhs(state, grid):
    """
    Linear stiff part L*q.
    Contains acoustic pressure gradient, buoyancy, base-state gradient.
    Used by SI (implicitly) and EPI (matrix exponential).

    Linear terms:
      u:  -cp * theta_bar * dpi'/dx
      w:  -cp * theta_bar * dpi'/dz  +  g * theta' / theta_bar
      pi: -(R/cv) * pi_bar * (du/dx + dw/dz)  +  gw/(cp*theta_bar)
      th: -w * dtheta_bar/dz
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

    # Broadcast 1D base-state profiles to 2D
    tb   = grid.theta_bar[:,    np.newaxis]   # (nz,1)
    pb   = grid.pi_bar[:,       np.newaxis]   # (nz,1)
    dtdz = grid.dtheta_bar_dz[:, np.newaxis]  # (nz,1)

    dpi_dx = _dx(pi, dx)
    dpi_dz = _dz(pi, dz)
    du_dx  = _dx(u,  dx)
    dw_dz  = _dz(w,  dz)

    rhs_u  = -cp * tb * dpi_dx
    rhs_w  = -cp * tb * dpi_dz + g * theta / tb
    rhs_pi = -(grid.Rd / cv) * pb * (du_dx + dw_dz) + g * w / (cp * tb)
    rhs_th = -w * dtdz

    # Boundary conditions
    rhs_w[0,  :] = 0.0
    rhs_w[-1, :] = 0.0
    rhs_th[0, :] = 0.0
    rhs_th[-1,:] = 0.0

    return {"u": rhs_u, "w": rhs_w, "theta": rhs_th, "pi": rhs_pi}


def compute_nonlinear_rhs(state, grid):
    """
    Nonlinear advection part N(q).
    Treated explicitly in all schemes.

    Nonlinear terms:
      u:  -u du/dx - w du/dz - cp*theta'*dpi'/dx
      w:  -u dw/dx - w dw/dz - cp*theta'*dpi'/dz
      pi: -u dpi'/dx - w dpi'/dz - (R/cv)*pi'*(du/dx+dw/dz)
      th: -u dtheta'/dx - w dtheta'/dz
    """
    u     = state["u"]
    w     = state["w"]
    theta = state["theta"]
    pi    = state["pi"]

    dx = grid.dx
    dz = grid.dz
    cp = grid.cp
    cv = grid.cv

    du_dx    = _dx(u,     dx)
    du_dz    = _dz(u,     dz)
    dw_dx    = _dx(w,     dx)
    dw_dz    = _dz(w,     dz)
    dth_dx   = _dx(theta, dx)
    dth_dz   = _dz(theta, dz)
    dpi_dx   = _dx(pi,    dx)
    dpi_dz   = _dz(pi,    dz)

    rhs_u  = -u*du_dx  - w*du_dz  - cp*theta*dpi_dx
    rhs_w  = -u*dw_dx  - w*dw_dz  - cp*theta*dpi_dz
    rhs_pi = -u*dpi_dx - w*dpi_dz - (grid.Rd/cv)*pi*(du_dx+dw_dz)
    rhs_th = -u*dth_dx - w*dth_dz

    rhs_w[0,  :] = 0.0
    rhs_w[-1, :] = 0.0
    rhs_th[0, :] = 0.0
    rhs_th[-1,:] = 0.0

    return {"u": rhs_u, "w": rhs_w, "theta": rhs_th, "pi": rhs_pi}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from grid import Grid

    print(f"\n  Numba active: {NUMBA_ACTIVE}")

    g     = Grid()
    state = g.allocate_state()

    # Zero amplitude test
    print("\n  Zero amplitude test (all values should be ~0):")
    for fn, label in [(compute_rhs, "full"),
                      (compute_linear_rhs, "linear"),
                      (compute_nonlinear_rhs, "nonlinear")]:
        rhs = fn(state, g)
        for k, v in rhs.items():
            mx = np.max(abs(v))
            status = "OK" if mx < 1e-15 else f"WARN: {mx:.2e}"
            print(f"    {label:>10}  {k:6s}  {status}")

    # Speed test
    import time
    print("\n  Speed test (100 RHS evaluations):")
    _ = compute_rhs(state, g)
    t0 = time.perf_counter()
    for _ in range(100):
        compute_rhs(state, g)
    elapsed = time.perf_counter() - t0
    print(f"    100 calls in {elapsed*1000:.1f} ms  ({elapsed*10:.2f} ms/call)")
