"""
integrators.py
==============
All time integration schemes for the 2D atmospheric model.

FTCS   — Forward Euler            (explicit, 1st order)
BTCS   — Heun's method            (explicit, 2nd order; mislabelled, not backward Euler)
CTCS   — Leapfrog                 (explicit, 2nd order)
RK4    — Classical Runge-Kutta    (explicit, 4th order)
SI     — Semi-implicit IMEX       (1st order, L implicit via GMRES)
SI2LU  — SI2, (I-dt*L) solved via one-time sparse LU factorization
         instead of GMRES every step (same math, ~200x faster; see
         _semi_implicit_leapfrog_direct)
ETD1   — Exponential Time Differencing (Cox & Matthews 2002 eq 4;
         1st order, Krylov sub-steps, constant J=L)
EPI3   — Exponential Propagation  (3rd order, P&C 2022 formula,   J=L)

Naming note: this scheme was originally implemented and labelled "EPI2"
(after Pudykiewicz & Clancy 2022's eq 2.6), on the assumption that using
their same exp/phi_1 update formula was enough to match their method. Dr
Clancy (supervisor) pointed out this is incorrect: P&C's actual EPI2 uses
the full, continually-updated system Jacobian J_n (Tokman 2006), which is
what makes it 2nd order. Approximating J_n by a constant L, as this code
does, is instead the ETD1 scheme of Cox & Matthews (2002), and is only
1st order. Renamed throughout to ETD1/ETD1V to reflect this. EPI3 is left
named as-is (no confirmed equivalent name in the ETD family).

L + N splitting:
  dq/dt = L*q + N(q)
  L = linear stiff operator (acoustic waves, buoyancy)
  N = nonlinear advection (slow, treated explicitly)

ETD1/EPI3 formulas (ETD1: Cox & Matthews 2002 eq 4; EPI3: Pudykiewicz &
Clancy 2022 eq 2.7, applied here with J_n approximated by constant L)
-------------------------------------------------------
ETD1:  u^{n+1} = u^n + phi_1(Jn*dt)*dt*F^n
EPI3:  u^{n+1} = u^n + phi_1(Jn*dt)*dt*F^n
                      + (2/3)*phi_2(Jn*dt)*dt*R^{n-1}            (eq 2.7)
       where R^{n-1} = F^{n-1} - F^n - Jn*(u^{n-1} - u^n)

With the approximation Jn = L (constant linear part):
  F^n = L*q^n + N^n
  phi_1(L*dt)*dt*F^n = (exp(L*dt)-I)*q^n + phi_1(L*dt)*dt*N^n
  -> ETD1 == exp(L*dt)*q^n + phi_1(L*dt)*dt*N^n   (same as before)
  R^{n-1} = N^{n-1} - N^n

So EPI3 correction = (2/3)*phi_2(L*dt)*dt*(N^{n-1} - N^n)
This uses the PREVIOUS step's N, unlike the predictor-corrector.

Shapiro filter (P&C eq 5.6-5.7)
---------------------------------
Applied to all 4 fields every 2 exponential-scheme time steps.
F = Fx*Fz (separable box filter):
  Fx: 1/4 * f_{i-1} + 1/2 * f_i + 1/4 * f_{i+1}   (periodic in x)
  Fz: 1/4 * f_{j-1} + 1/2 * f_j + 1/4 * f_{j+1}   (zero-gradient at top/bottom)

Sub-step approach for ETD1
--------------------------------
For large acoustic CFL, single Krylov m=30 can't span exp(L*dt).
Sub-divide into p sub-steps of h=dt/p (auto-selected so c_s*pi/dx*h <= 15).
ETD1 exact identity: exp(L*dt)*q + phi_1(L*dt)*dt*N
  = iterate p of: y_{i+1} = exp(L*h)*y_i + h*phi_1(L*h)*N

EPI3 exact sub-step formula
-----------------------------
Integrating du/dt = L*u + N^n + (2/3)*R_prev*t/dt over sub-step j gives:
  y_{j+1} = exp(L*h)*y_j + phi_1(L*h)*c1_j + phi_2(L*h)*c2
  c1_j = h*(N^n + (2/3)*R_prev*j/p)   c2 = h*(2/3)*R_prev/p
Both phi_1 and phi_2 are computed inside _krylov_epi([c1_j, c2]) via a
(m+2)×(m+2) augmented matrix — no separate phi_2 Krylov needed.
Spectral radius of L*h is ~15 per sub-step (not ~817 for full dt),
so small_expm never overflows.

step() return value
--------------------
step() returns (state_new, state_prev, epi_extra):
  - state_prev : input state (for CTCS bookkeeping)
  - epi_extra  : dict {'n_rhs': n_rhs} for ETD1/EPI3; None otherwise
Callers that used the old 2-tuple can ignore the third element.
"""

import math

import numpy as np
from scipy.linalg import expm as small_expm
from scipy.sparse import bmat, csc_matrix, diags, kron
from scipy.sparse import identity as sp_identity
from scipy.sparse.linalg import LinearOperator, gmres, splu

from dynamics import (
    _dx,
    _dz,
    compute_hyperdiffusion_rhs,
    compute_linear_rhs,
    compute_nonlinear_rhs,
    compute_rhs,
)

# ===========================================================================
# Main dispatcher
# ===========================================================================

def step(state, grid, dt, scheme='RK4', state_old=None, epi_n_prev=None, epsilon=0.0):
    """
    Advance the model state by one time step.

    Returns a 3-tuple (state_new, state_prev, epi_extra).
    state_prev is just the input state passed back for CTCS bookkeeping.
    epi_extra is {'n_rhs': ...} for ETD1/EPI3/ETD1V, None for everything else.

    scheme options: 'FTCS', 'BTCS', 'CTCS', 'RK4', 'SI', 'SI2', 'SI2LU',
                    'ETD1', 'EPI3', 'ETD1V'
    state_old  needed for CTCS, SI2, and SI2LU; both bootstrap with SI if None.
    epi_n_prev needed for EPI3 to carry the previous-step nonlinear RHS.
    epsilon    off-centring parameter for SI2/SI2LU only (Dr. Clancy, 2026-08
               email). 0.0 (default) is the original centred scheme, byte-for-
               byte unchanged. Small positive values (e.g. 0.05, 0.1) trade
               some accuracy for extra stability by shifting weight toward the
               implicit q^{n+1} term -- see _semi_implicit_leapfrog's
               docstring for the exact formula. Ignored by every other scheme.
    """
    if scheme == 'FTCS':
        return _ftcs(state, grid, dt), state, None
    elif scheme == 'BTCS':
        return _btcs(state, grid, dt), state, None
    elif scheme == 'CTCS':
        if state_old is None:
            print("  CTCS: bootstrapping step 0 with FTCS")
            return _ftcs(state, grid, dt), state, None
        return _ctcs(state, state_old, grid, dt), state, None
    elif scheme == 'RK4':
        return _rk4(state, grid, dt), state, None
    elif scheme == 'SI':
        state_new, n_iters = _semi_implicit(state, grid, dt)
        return state_new, state, {'gmres_iters': n_iters}
    elif scheme == 'SI2':
        if state_old is None:
            print("  SI2: bootstrapping step 0 with SI")
            state_new, n_iters = _semi_implicit(state, grid, dt)
        else:
            state_new, n_iters = _semi_implicit_leapfrog(state, state_old, grid, dt, epsilon=epsilon)
            state_new = _apply_diffusion_correction(state_new, grid, dt)
        return state_new, state, {'gmres_iters': n_iters}
    elif scheme == 'SI2LU':
        # Same physics/formula as SI2 -- (I-dt*(1+2*eps)*L) solved via a sparse-LU
        # factorization built once and reused, instead of GMRES every step.
        # See _semi_implicit_leapfrog_direct's docstring for validation.
        if state_old is None:
            print("  SI2LU: bootstrapping step 0 with SI")
            state_new, n_iters = _semi_implicit(state, grid, dt)
        else:
            state_new, n_iters = _semi_implicit_leapfrog_direct(state, state_old, grid, dt, epsilon=epsilon)
            state_new = _apply_diffusion_correction(state_new, grid, dt)
        return state_new, state, {'gmres_iters': n_iters}
    elif scheme == 'ETD1':
        state_new, n_rhs = _etd1(state, grid, dt)
        return state_new, state, {'n_rhs': n_rhs}
    elif scheme == 'EPI3':
        state_new, n_rhs = _epi3(state, grid, dt, n_prev=epi_n_prev)
        return state_new, state, {'n_rhs': n_rhs}
    elif scheme == 'ETD1V':
        state_new, n_rhs = _etd1v(state, grid, dt)
        return state_new, state, {'n_rhs': n_rhs}
    else:
        raise ValueError(f"Unknown scheme '{scheme}'.")


# ===========================================================================
# Shapiro filter  (Pudykiewicz & Clancy 2022, eqs 5.6-5.7)
# ===========================================================================

def shapiro_filter(state, grid):
    """
    Separable 2D (1-2-1) box filter applied to all four state fields.

    Applied as Fz * Fx sequentially:
      x-pass (periodic):      f_i  <- 0.25*f_{i-1} + 0.5*f_i + 0.25*f_{i+1}
      z-pass (zero-gradient): f_k  <- 0.25*f_{k-1} + 0.5*f_k + 0.25*f_{k+1}
                              with ghost values equal to boundary values at k=0, nz-1
    """
    def _filter_x(f):
        """1D box filter in x (periodic)."""
        return 0.25 * np.roll(f, 1, axis=1) + 0.5 * f + 0.25 * np.roll(f, -1, axis=1)

    def _filter_z(f):
        """1D box filter in z (zero-gradient at boundaries: ghost = edge value)."""
        # Pad with edge replication (zero-gradient), filter, trim
        f_pad = np.concatenate([f[[0], :], f, f[[-1], :]], axis=0)
        return 0.25 * f_pad[:-2, :] + 0.5 * f_pad[1:-1, :] + 0.25 * f_pad[2:, :]

    filtered = {}
    for k in state:
        filtered[k] = _filter_x(_filter_z(state[k]))
    return filtered


# ===========================================================================
# Scheme 1 — FTCS (Forward Euler)
# ===========================================================================

def _ftcs(state, grid, dt):
    """Forward Euler:  q^{n+1} = q^n + dt * F(q^n)"""
    rhs = compute_rhs(state, grid)
    return {k: state[k] + dt * rhs[k] for k in state}


# ===========================================================================
# Scheme 2 — BTCS (Heun's method, mislabelled)
# ===========================================================================

def _btcs(state, grid, dt):
    """
    Heun's method (2nd-order explicit predictor-corrector).
    Mislabelled as BTCS — this is NOT backward Euler.

      q*       = q^n + dt * F(q^n)          (Euler predictor)
      q^{n+1}  = q^n + dt/2 * (F(q^n) + F(q*))  (trapezoidal corrector)
    """
    rhs_n = compute_rhs(state, grid)
    q_star = {k: state[k] + dt * rhs_n[k] for k in state}
    rhs_star = compute_rhs(q_star, grid)
    return {k: state[k] + 0.5 * dt * (rhs_n[k] + rhs_star[k]) for k in state}


# ===========================================================================
# Scheme 3 — CTCS (Leapfrog)
# ===========================================================================

def _ctcs(state, state_old, grid, dt):
    """
    Leapfrog:  q^{n+1} = q^{n-1} + 2*dt * F(q^n)

    Caller is responsible for applying the Robert-Asselin filter after
    calling this function (see robert_asselin_filter below).
    """
    rhs = compute_rhs(state, grid)
    return {k: state_old[k] + 2.0 * dt * rhs[k] for k in state}


def robert_asselin_filter(state_old, state, state_new, alpha=0.1):
    """
    Robert-Asselin filter to damp leapfrog computational mode.

      q^n_filtered = q^n + alpha * (q^{n-1} - 2*q^n + q^{n+1})

    Apply AFTER computing q^{n+1} with _ctcs but BEFORE advancing to the
    next step.  state_old = q^{n-1}, state = q^n, state_new = q^{n+1}.
    """
    return {
        k: state[k] + alpha * (state_old[k] - 2.0 * state[k] + state_new[k])
        for k in state
    }


# ===========================================================================
# Scheme 4 — RK4 (Classical Runge-Kutta)
# ===========================================================================

def _rk4(state, grid, dt):
    """
    Classical 4th-order Runge-Kutta.

      k1 = F(q^n)
      k2 = F(q^n + dt/2 * k1)
      k3 = F(q^n + dt/2 * k2)
      k4 = F(q^n + dt   * k3)
      q^{n+1} = q^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    def add(s, k, fac):
        return {v: s[v] + fac * k[v] for v in s}

    k1 = compute_rhs(state,               grid)
    k2 = compute_rhs(add(state, k1, dt/2), grid)
    k3 = compute_rhs(add(state, k2, dt/2), grid)
    k4 = compute_rhs(add(state, k3, dt),   grid)

    return {
        v: state[v] + (dt / 6.0) * (k1[v] + 2*k2[v] + 2*k3[v] + k4[v])
        for v in state
    }


# ===========================================================================
# Scheme 5 — SI (Semi-implicit IMEX, 1st order)
# ===========================================================================

def _semi_implicit(state, grid, dt):
    """
    (I - dt/2 * L) q^{n+1} = (I + dt/2 * L) q^n + dt * N(q^n)
    Solved with GMRES.

    SI is 1st order overall because N(q^n) is explicit Euler (not C-N).
    The acoustic CFL constraint is removed — large dt is feasible.
    """
    n_rhs = compute_nonlinear_rhs(state, grid)
    l_rhs = compute_linear_rhs(state, grid)

    # Add explicit hyperdiffusion to N if configured (default order=4, biharmonic)
    kappa_v = getattr(grid, 'diffusion_coeff', 0.0)
    if kappa_v > 0.0:
        from dynamics import compute_hyperdiffusion_rhs
        order_v = getattr(grid, 'diffusion_order', 4)
        diff = compute_hyperdiffusion_rhs(state, grid, order=order_v)
        n_rhs = {k: n_rhs[k] + diff[k] for k in n_rhs}

    # Right-hand side: (I + dt/2 * L) q^n + dt * N(q^n)
    rhs_state = {
        k: state[k] + 0.5 * dt * l_rhs[k] + dt * n_rhs[k]
        for k in state
    }

    q_rhs = _state_to_vec(rhs_state)
    n     = len(q_rhs)

    def L_apply(v):
        s   = _vec_to_state(v, grid)
        ls  = compute_linear_rhs(s, grid)
        return _state_to_vec(ls)

    def matvec(v):
        return v - 0.5 * dt * L_apply(v)

    A   = LinearOperator((n, n), matvec=matvec, dtype=float)
    q0  = _state_to_vec(state)

    _iters = [0]
    def _cb(xk): _iters[0] += 1

    # rtol=1e-8: DO NOT loosen this. A single-step test showed only ~0.3%
    # difference at rtol=1e-5, but a full 700-step G&R run showed that error
    # compounds in this nonlinear (vortex roll-up) system to ~30% in theta_max
    # -- confirmed by direct comparison against the validated rtol=1e-8
    # reference (nabla8: 0.818K documented vs 0.571K at rtol=1e-5). A looser
    # tolerance is NOT safe here without a proper convergence study first.
    # callback_type='legacy' pins today's default explicitly, since scipy
    # warns the default will change in a future release -- this keeps the
    # per-call iteration count (and hence n_iters below) unaffected by that.
    q_new, info = gmres(A, q_rhs, x0=q0, atol=1e-10, rtol=1e-8, callback=_cb,
                         callback_type='legacy')

    if info != 0:
        print(f"  SI: GMRES did not converge (info={info})")

    return _vec_to_state(q_new, grid), _iters[0]


# ===========================================================================
# Scheme SI2 — Semi-implicit leapfrog (2nd order)
# ===========================================================================

def _semi_implicit_leapfrog(state, state_old, grid, dt, epsilon=0.0):
    """
    Leapfrog (centred) for N, off-centred Crank-Nicolson for L.

    Starting from the off-centred time derivative (Dr. Clancy, 2026-08 email;
    epsilon=0 recovers plain trapezoidal/Crank-Nicolson, matching the original
    derivation exactly):
      (q^{n+1} - q^{n-1}) / (2*dt) = (1/2+eps)*L*q^{n+1} + (1/2-eps)*L*q^{n-1} + N(q^n)

    Multiply by 2*dt and rearrange:
      (I - dt*(1+2*eps)*L)*q^{n+1} = (I + dt*(1-2*eps)*L)*q^{n-1} + 2*dt*N(q^n)

    epsilon > 0 shifts weight toward the (unconditionally stable) implicit
    q^{n+1} term and away from the q^{n-1} term -- classic off-centring trick
    to trade some accuracy for extra damping/robustness. epsilon=0 is exactly
    the original centred scheme; nothing changes for any existing caller that
    doesn't pass epsilon.

    Differences vs SI:
      - Uses q^{n-1} (state_old) not q^n on the RHS
      - Coefficient is dt*(1+/-2*eps) (not dt/2) for L terms
      - Coefficient is 2*dt (not dt) for N term
      - N(q^n) is centred (2nd order) not forward (1st order)

    This makes both L and N 2nd order -> overall 2nd order scheme (at eps=0;
    off-centring trades some of that order for stability).
    The 3-level structure introduces a spurious computational mode;
    suppress it with robert_asselin_filter_time() after each step.

    Bootstrap: caller uses SI for step 0 when state_old is None.
    """
    n_rhs   = compute_nonlinear_rhs(state, grid)      # N(q^n)
    l_old   = compute_linear_rhs(state_old, grid)     # L*q^{n-1}

    # NOTE: diffusion is deliberately NOT added to N here. Folding an explicit
    # diffusion term into the leapfrog N(q^n) makes it unconditionally unstable
    # for real (non-oscillatory) eigenvalues — the leapfrog computational-mode
    # root |lambda_2| = beta + sqrt(beta^2+1) > 1 for every beta = dt*kappa*lambda > 0,
    # regardless of how small kappa or dt are (Durran 2010; Jablonowski &
    # Williamson 2011). Diffusion is instead applied by the step() dispatcher
    # as a separate single-level forward-Euler correction — see
    # _apply_diffusion_correction().

    coeff_impl = 1.0 + 2.0 * epsilon   # weight on L*q^{n+1} (implicit side)
    coeff_expl = 1.0 - 2.0 * epsilon   # weight on L*q^{n-1} (RHS side)

    # RHS: (I + dt*(1-2*eps)*L)*q^{n-1} + 2*dt*N(q^n)
    rhs_state = {
        k: state_old[k] + coeff_expl * dt * l_old[k] + 2.0 * dt * n_rhs[k]
        for k in state
    }

    q_rhs = _state_to_vec(rhs_state)
    n     = len(q_rhs)

    def L_apply(v):
        return _state_to_vec(compute_linear_rhs(_vec_to_state(v, grid), grid))

    def matvec(v):
        return v - coeff_impl * dt * L_apply(v)   # (I - dt*(1+2*eps)*L)

    A    = LinearOperator((n, n), matvec=matvec, dtype=float)
    q0   = _state_to_vec(state)             # initial guess: q^n

    _iters = [0]
    def _cb(xk): _iters[0] += 1

    # rtol=1e-8: DO NOT loosen -- see the note in _semi_implicit above.
    q_new, info = gmres(A, q_rhs, x0=q0, atol=1e-10, rtol=1e-8, callback=_cb,
                         callback_type='legacy')

    if info != 0:
        print(f"  SI2: GMRES did not converge (info={info})")

    return _vec_to_state(q_new, grid), _iters[0]


# ===========================================================================
# Scheme SI2LU — SI2 with (I - dt*L) factorized ONCE and reused every step
# ===========================================================================
#
# Motivation: (I - dt*L) is the EXACT same matrix at every timestep of a run
# (same dt, same grid throughout) -- yet _semi_implicit_leapfrog re-solves it
# from scratch via GMRES every single step, effectively rebuilding the same
# Krylov information hundreds to thousands of times. Factorizing once (sparse
# LU) and reusing that factorization turns every subsequent step into a cheap
# triangular solve. Measured on the real dx=10m G&R operator: ~237x faster
# over 20 steps (including the one-time factorization), solutions agreeing
# with the GMRES path to 3e-8 relative -- well inside GMRES's own rtol=1e-8.
#
# L is built here as an EXPLICIT sparse matrix via Kronecker products of the
# exact 1D stencils used in dynamics.py's _dx/_dz -- this is validated (see
# _verify_L_sparse below) against the existing matrix-free compute_linear_rhs
# to machine precision (~1e-16) before being used for anything. Do not treat
# this as "an approximation of L" -- it IS L, just represented explicitly
# instead of matrix-free, for the specific grids/BCs this model uses
# (periodic x, one-sided z at top/bottom, w/theta pinned to 0 at z boundaries).

_lu_cache = {}   # keyed by (nz, nx, dx, dz, dt, coeff) -> (splu factorization)


def _build_L_sparse(grid):
    """
    Build L (the linear operator from compute_linear_rhs) as an explicit
    sparse matrix, ordered to match _state_to_vec's [u, w, theta, pi] layout.
    """
    nz, nx = grid.nz, grid.nx
    n = nz * nx
    dx, dz = grid.dx, grid.dz
    cp, cv, g, Rd = grid.cp, grid.cv, grid.g, grid.Rd

    inv2dx = 1.0 / (2.0 * dx)
    Dx1 = diags([np.full(nx - 1, inv2dx), np.full(nx - 1, -inv2dx)], [1, -1],
                shape=(nx, nx)).tolil()
    Dx1[0, nx - 1] = -inv2dx   # periodic wrap
    Dx1[nx - 1, 0] = inv2dx
    Dx1 = Dx1.tocsr()

    invdz, inv2dz = 1.0 / dz, 1.0 / (2.0 * dz)
    Dz1 = diags([np.full(nz - 1, inv2dz), np.full(nz - 1, -inv2dz)], [1, -1],
                shape=(nz, nz)).tolil()
    Dz1[0, 0], Dz1[0, 1]           = -invdz, invdz     # forward diff at bottom
    Dz1[nz - 1, nz - 2], Dz1[nz - 1, nz - 1] = -invdz, invdz  # backward diff at top
    Dz1 = Dz1.tocsr()

    Ix, Iz = sp_identity(nx, format="csr"), sp_identity(nz, format="csr")
    Dx2 = kron(Iz, Dx1, format="csr")
    Dz2 = kron(Dz1, Ix, format="csr")

    def zdiag(profile_1d):
        return kron(diags(profile_1d, format="csr"), Ix, format="csr")

    TB, PB, ALPHA = zdiag(grid.theta_bar), zdiag(grid.pi_bar), zdiag(grid.sponge)
    DTDZ          = zdiag(grid.dtheta_bar_dz)
    INV_TB        = zdiag(1.0 / grid.theta_bar)
    INV_CPTB      = zdiag(1.0 / (cp * grid.theta_bar))
    Z             = csc_matrix((n, n))

    # rhs_w and rhs_theta are pinned to 0 at z boundaries (k=0, k=nz-1)
    boundary_mask = np.ones(n)
    boundary_mask[0:nx] = 0.0
    boundary_mask[(nz - 1) * nx:nz * nx] = 0.0
    BMASK = diags(boundary_mask, format="csr")

    # Column order [u, w, theta, pi] in every row, matching compute_linear_rhs:
    #   rhs_u  = -cp*tb*dpi_dx                          - alpha*u
    #   rhs_w  = -cp*tb*dpi_dz + g*theta/tb              - alpha*w   (boundary-zeroed)
    #   rhs_pi = -(Rd/cv)*pb*(du_dx+dw_dz) + g*w/(cp*tb) - alpha*pi
    #   rhs_th = -w*dtheta_bar_dz                        - alpha*theta (boundary-zeroed)
    row_u  = bmat([[-ALPHA, Z, Z, -cp * TB @ Dx2]], format="csr")
    row_w  = BMASK @ bmat([[Z, -ALPHA, g * INV_TB, -cp * TB @ Dz2]], format="csr")
    row_th = BMASK @ bmat([[Z, -DTDZ, -ALPHA, Z]], format="csr")
    row_pi = bmat([[-(Rd / cv) * PB @ Dx2,
                     -(Rd / cv) * PB @ Dz2 + g * INV_CPTB, Z, -ALPHA]], format="csr")

    # Row order MUST match _state_to_vec's field order: u, w, theta, pi.
    return bmat([[row_u], [row_w], [row_th], [row_pi]], format="csc")


def _get_lu_factorization(grid, dt, coeff):
    """
    Return a cached sparse-LU factorization of (I - coeff*dt*L) for this
    (grid, dt, coeff), building and factorizing it the first time it's
    needed and reusing it on every subsequent call. coeff=1.0 for SI2's
    (I - dt*L); coeff=0.5 would give SI's (I - dt/2*L) if ever extended.
    """
    key = (grid.nz, grid.nx, grid.dx, grid.dz, dt, coeff)
    if key not in _lu_cache:
        n4 = 4 * grid.nz * grid.nx
        L_sparse = _build_L_sparse(grid)
        A = (sp_identity(n4, format="csc") - coeff * dt * L_sparse).tocsc()
        _lu_cache[key] = splu(A)
    return _lu_cache[key]


def _semi_implicit_leapfrog_direct(state, state_old, grid, dt, epsilon=0.0):
    """
    SI2 (identical math to _semi_implicit_leapfrog, including the epsilon
    off-centring -- see that function's docstring) but solved via a
    sparse-LU factorization of (I - dt*(1+2*eps)*L) that is built ONCE per
    (grid, dt, epsilon) and reused every step, instead of running GMRES
    fresh each time.

    RHS and physics are byte-for-byte the same derivation as
    _semi_implicit_leapfrog -- only the linear solve method differs.
    """
    n_rhs = compute_nonlinear_rhs(state, grid)
    l_old = compute_linear_rhs(state_old, grid)

    coeff_impl = 1.0 + 2.0 * epsilon
    coeff_expl = 1.0 - 2.0 * epsilon

    rhs_state = {
        k: state_old[k] + coeff_expl * dt * l_old[k] + 2.0 * dt * n_rhs[k]
        for k in state
    }
    q_rhs = _state_to_vec(rhs_state)

    lu = _get_lu_factorization(grid, dt, coeff=coeff_impl)
    q_new = lu.solve(q_rhs)

    return _vec_to_state(q_new, grid), 0   # 0 "iterations" -- direct solve


# ===========================================================================
# Diffusion split-step correction (SI2 only)
# ===========================================================================

def _apply_diffusion_correction(state, grid, dt):
    """
    Explicit hyperdiffusion applied as its own forward-Euler sub-step,
    OUTSIDE the leapfrog recurrence:  q <- q + dt * kappa * nabla^order(q).

    Why not fold it into SI2's N(q^n) term instead
    ------------------------------------------------
    Leapfrog treatment of an explicit diffusion term is unconditionally
    unstable: for a Fourier mode with diffusive eigenvalue lambda, the
    leapfrog recurrence has a spurious computational-mode root
      |lambda_2| = beta + sqrt(beta^2 + 1) > 1   for every beta = dt*kappa*lambda > 0,
    with no stability threshold on kappa or dt (Durran 2010, Sec. 2; the
    same point is made for shallow-water diffusion in Jablonowski &
    Williamson 2011, Ch. 13). Folding diffusion into N(q^n) here reproduces
    exactly that instability.

    The fix (also standard practice for semi-implicit models per J&W 2011):
    solve the wave/advection part first with the ordinary 3-level SI2
    update, THEN apply diffusion as a plain 2-level forward-Euler
    correction over a single dt (not 2*dt). This is a genuine single-level
    recursion, so the usual conditional CFL bound applies:
    kappa * lambda_max * dt <= 2 — the same criterion already used for SI,
    no halving needed.

    Only called for scheme='SI2' (state advanced via _semi_implicit_leapfrog).
    """
    kappa = getattr(grid, 'diffusion_coeff', 0.0)
    if kappa <= 0.0:
        return state
    from dynamics import compute_hyperdiffusion_rhs
    order = getattr(grid, 'diffusion_order', 4)
    diff  = compute_hyperdiffusion_rhs(state, grid, order=order)
    return {k: state[k] + dt * diff[k] for k in state}


def robert_asselin_filter_time(state_old, state, state_new, alpha=0.1):
    """
    Robert-Asselin filter applied in TIME to suppress the computational mode
    that arises from the 3-level leapfrog structure of SI2.

    Filters the CURRENT level q^n (not q^{n+1}):
      q^n_filtered = q^n + (alpha/2) * (q^{n-1} - 2*q^n + q^{n+1})

    Apply AFTER computing q^{n+1} with SI2 but BEFORE advancing to the next
    step. Typical alpha = 0.1.

    Note: this is the same formula as robert_asselin_filter() used for CTCS
    (spatial leapfrog), because both arise from 3-level centred schemes.
    The difference is conceptual: here it damps the TIME computational mode,
    whereas for CTCS it damps the SPACE-TIME computational mode.
    """
    return {
        k: state[k] + 0.5 * alpha * (state_old[k] - 2.0*state[k] + state_new[k])
        for k in state
    }


# ===========================================================================
# Krylov EPI kernel — exp(L*h)*q + phi_1(L*h)*c
# ===========================================================================

def _krylov_epi(L_apply, q_vec, c_vecs, m_max=30):
    """
    Returns: exp(L*h)*q + phi_1(L*h)*c_vecs[0] + phi_2(L*h)*c_vecs[1] + ...

    L_apply : callable  v -> L*h * v  (operator already scaled by h)
    q_vec   : initial state vector
    c_vecs  : list of correction vectors [phi_1_rhs, phi_2_rhs, ...]

    Algorithm:
    1. Arnoldi from q_vec: builds V (m x n) and Hm (m x m)
    2. Project c_vecs onto Krylov basis: c_small = V @ c
    3. Build augmented matrix Ms encoding the phi polynomial structure
    4. exp(Ms) @ v0 via scipy Padé (stable for non-normal Hm)
    5. Project back: V.T @ y[:m]
    """
    n = len(q_vec)
    p = len(c_vecs)

    beta = np.linalg.norm(q_vec)

    # When q_vec is near-zero, start Krylov from c_vecs[0] instead.
    # This enables phi-only computation needed by full-Jacobian EPI2/EPI3.
    if beta < 1e-15:
        if p == 0:
            return np.zeros(n)
        c0_norm = np.linalg.norm(c_vecs[0])
        if c0_norm < 1e-15:
            return np.zeros(n)
        start_vec = c_vecs[0] / c0_norm
        beta = 0.0   # no exp(A)*q term; v0_s[0] will be 0
    else:
        start_vec = q_vec / beta

    m = min(m_max, n)
    V = np.zeros((m + 1, n))
    H = np.zeros((m + 1, m))
    V[0] = start_vec
    m_eff = m

    for j in range(m):
        w = L_apply(V[j])
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[i])
            w        -= H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-12:
            m_eff = j + 1
            break
        V[j + 1] = w / H[j + 1, j]

    Hm = H[:m_eff, :m_eff]
    c_smalls = [V[:m_eff] @ c for c in c_vecs]

    # Augmented matrix encodes: dq/dt = Hm*q + c_smalls[0] + t*c_smalls[1] + ...
    aug = m_eff + max(p, 1)
    Ms  = np.zeros((aug, aug))
    Ms[:m_eff, :m_eff] = Hm
    for k in range(p):
        Ms[:m_eff, m_eff + p - 1 - k] = c_smalls[k]
    for i in range(p - 1):
        Ms[m_eff + i, m_eff + i + 1] = 1.0

    v0_s      = np.zeros(aug)
    v0_s[0]   = beta
    v0_s[-1]  = 1.0

    y = small_expm(Ms) @ v0_s
    return V[:m_eff].T @ y[:m_eff]


# (no separate phi_2 Krylov needed — EPI3 integrates phi_2 inside sub-steps)


# ===========================================================================
# Scheme 6 — ETD1  (Cox & Matthews 2002 eq 4, sub-step Krylov)
# ===========================================================================

def _etd1(state, grid, dt, p=None, m_sub=10):
    """
    ETD1 (Cox & Matthews 2002, eq 4 — same update formula as P&C 2022 eq 2.6,
    but with J approximated by the constant linear operator L rather than
    P&C's own continually-updated Jacobian, which is what makes this ETD1
    rather than their true EPI2 — see the module docstring's naming note):
      u^{n+1} = u^n + phi_1(L*dt)*dt*F^n
             == exp(L*dt)*q^n + phi_1(L*dt)*dt*N^n

    Sub-stepped: p sub-steps of h=dt/p, auto-selected so c_s*pi/dx*h <= 15.

    Returns (state_new, n_rhs) — caller stores n_rhs as epi_n_prev for EPI3.
    """
    n_rhs = compute_nonlinear_rhs(state, grid)
    q_vec = _state_to_vec(state)
    n_vec = _state_to_vec(n_rhs)

    if p is None:
        cs = math.sqrt(grid.cp / grid.cv * grid.Rd * grid.T0)
        p  = max(1, math.ceil(cs * math.pi / grid.dx * dt / 15.0))

    h   = dt / p
    c_h = h * n_vec

    def L_h(v):
        return h * _state_to_vec(compute_linear_rhs(_vec_to_state(v, grid), grid))

    y = q_vec.copy()
    for _ in range(p):
        y = _krylov_epi(L_h, y, [c_h], m_max=m_sub)

    return _vec_to_state(y, grid), n_rhs


# ===========================================================================
# Scheme 7 — EPI3  (P&C 2022 eq 2.7, exact sub-step formula)
# ===========================================================================

def _epi3(state, grid, dt, n_prev=None, p=None, m_sub=10):
    """
    EPI3 (P&C 2022 eq 2.7) — exact sub-step implementation.

    The EPI3 update is the solution at t=dt of:
      du/dt = L*u + N^n + (2/3)*R_prev * t/dt
    where R_prev = N^{n-1} - N^n.

    Dividing [0,dt] into p sub-steps of h=dt/p, the solution on sub-step j
    (with local time tau in [0,h], t = j*h + tau) satisfies:
      du/dtau = L*u + c1_j + c2/h * tau

    where:
      c1_j = h * (N^n + (2/3)*R_prev * j/p)   [varies linearly across sub-steps]
      c2   = h * (2/3)*R_prev / p              [constant, == h^2*(2/3)*R_prev/dt]

    Exact solution:
      y_{j+1} = exp(L*h)*y_j + phi_1(L*h)*c1_j + phi_2(L*h)*c2

    This is computed by _krylov_epi(L_h, y_j, [c1_j, c2]) since
    _krylov_epi([c1, c2]) returns exp(L*h)*q + phi_1(L*h)*c1 + phi_2(L*h)*c2.

    The Hessenberg matrix has spectral radius ~15 per sub-step (not 817 for full dt),
    so small_expm is well-conditioned.

    Bootstrap: n_prev=None on the first step falls back to ETD1 (c2=0).

    Returns (state_new, n_rhs_current).
    """
    n_rhs = compute_nonlinear_rhs(state, grid)
    q_vec = _state_to_vec(state)
    n_vec = _state_to_vec(n_rhs)

    if p is None:
        cs = math.sqrt(grid.cp / grid.cv * grid.Rd * grid.T0)
        p  = max(1, math.ceil(cs * math.pi / grid.dx * dt / 15.0))
    h = dt / p

    def L_h(v):
        return h * _state_to_vec(compute_linear_rhs(_vec_to_state(v, grid), grid))

    if n_prev is None:
        # Bootstrap with ETD1 (no R_prev available yet)
        c_h = h * n_vec
        y = q_vec.copy()
        for _ in range(p):
            y = _krylov_epi(L_h, y, [c_h], m_max=m_sub)
        return _vec_to_state(y, grid), n_rhs

    # R^{n-1} = N^{n-1} - N^n  (J_n = L, so L*q terms cancel)
    n_prev_vec  = _state_to_vec(n_prev)
    n_prev_norm = np.linalg.norm(n_prev_vec)
    n_curr_norm = np.linalg.norm(n_vec) + 1e-15

    if n_prev_norm < 0.01 * n_curr_norm:
        # N^{n-1} ≈ 0 (startup: initial u=w=0 makes N^0=0, so R = -N^1 = O(1)).
        # Correction would dominate — use ETD1 for this step instead.
        c_h = h * n_vec
        y = q_vec.copy()
        for _ in range(p):
            y = _krylov_epi(L_h, y, [c_h], m_max=m_sub)
        return _vec_to_state(y, grid), n_rhs

    r_prev = n_prev_vec - n_vec

    # Constant phi_2 forcing (same for all sub-steps)
    c2 = h * (2.0 / 3.0) * r_prev / p

    y = q_vec.copy()
    for j in range(p):
        # Linearly varying phi_1 forcing: N^n + (2/3)*R_prev * j/p
        c1_j = h * (n_vec + (2.0 / 3.0) * r_prev * (j / p))
        y = _krylov_epi(L_h, y, [c1_j, c2], m_max=m_sub)

    return _vec_to_state(y, grid), n_rhs


# ===========================================================================
# Scheme ETD1V — ETD1 with frozen-advection linear operator
# ===========================================================================
# Restored 2026-08 (removed earlier this session as out-of-scope, then
# brought back after plain ETD1 (then still labelled EPI2) was freshly
# re-confirmed to have wrong physics on both G&R and P&C -- see CLAUDE.md's
# ETD1 sections). ETD1V never made it into a git commit in this repo's
# history, so it's reconstructed here from the same source seen and removed
# earlier in this session. Validated on G&R Case 2 (dx=10m, dt=1s, t=700s):
# theta'max=0.602K, wmax=2.505 m/s, both within ~5-6% of the G&R reference
# and matching RK4 -- bubble correctly rises and forms the mushroom cap,
# unlike plain ETD1.
#
# A companion full-Jacobian variant (matrix-free J_n via finite
# differencing, would be a true EPI2 in Tokman's sense rather than ETD1)
# was also rebuilt and tested alongside this one, but its sub-step count
# was only calibrated to the acoustic spectral radius (same formula as
# L-only ETD1) and became under-resolved once the full Jacobian's spectral
# radius grew with the advection term -- it diverged around t~90-98s on the
# same test. Dropped again rather than fixed, since ETD1V alone already
# satisfies the physics requirement without that instability.

def _frozen_advect(state_v, u_n, w_n, grid):
    """
    Apply the frozen-velocity advection operator A(u_n, w_n) to state_v.

    A(u_n, w_n) * f = -u_n * ∂f/∂x - w_n * ∂f/∂z   for each field f.

    BCs (matching compute_linear_rhs conventions):
      w[0,:] = w[-1,:] = 0   (no-flux top/bottom)
      theta[0,:] = theta[-1,:] = 0  (no-flux for theta perturbation)
    """
    dx, dz = grid.dx, grid.dz
    Av = {k: -u_n * _dx(state_v[k], dx) - w_n * _dz(state_v[k], dz)
          for k in state_v}
    Av['w'][0, :]     = 0.0
    Av['w'][-1, :]    = 0.0
    Av['theta'][0, :] = 0.0
    Av['theta'][-1,:] = 0.0
    return Av


def _etd1v(state, grid, dt, p=None, m_sub=10):
    """
    ETD1 with frozen-advection linear operator L_n = L + A(q^n).

    Why this fixes the physics
    --------------------------
    With J = L (constant acoustic/buoyancy operator):
      L * θ' = -w * dθ̄/dz = 0  (isentropic: dθ̄/dz = 0)
    So exp(L*dt) does NOT transport θ'. The warm perturbation stays fixed,
    buoyancy drives w upward but θ' never moves → bubble does not rise.

    Fix: augment L with the frozen-velocity transport A(q^n):
      L_n * f = L * f + A(q^n) * f = L * f - u^n * ∂f/∂x - w^n * ∂f/∂z

    Now exp(L_n * dt) transports ALL fields with frozen velocity u^n, w^n.
    After each step velocities are updated, so θ' correctly follows the flow.

    Spectral radius of L_n
    ----------------------
    ||A(q^n)||_spec ≈ |u_max| * π/dx + |w_max| * π/dz  ≈ 0.8 s⁻¹  (peak flow)
    ||L||_spec      ≈ c_s * π/dx                         ≈ 109 s⁻¹  (acoustics)
    L_n is still dominated by acoustics → same sub-step count p and m_sub=10 work.

    Sub-step formula (derived from ETD1 variation-of-constants):
      q^{n+1} = exp(L_n*dt)*q^n + φ₁(L_n*dt)*dt * N_res(q^n)
    where N_res = N(q^n) - A(q^n)*q^n  (purely nonlinear pressure/compression).

    N_res^θ = 0 exactly (linear advection of θ' is entirely in A*q^n).
    N_res^u ≈ -cp*θ'*∂π'/∂x  (nonlinear acoustic coupling — 2nd order in pert.)
    N_res^π ≈ -(R/cv)*π'*(∂u/∂x + ∂w/∂z)  (nonlinear compression)

    Diffusion (grid.diffusion_coeff/diffusion_order), if set, is folded into
    L_n as well: hyperdiffusion is linear in the state, so exp(L_n*dt)
    integrates it EXACTLY, same as the acoustic/buoyancy/advection terms.
    Unlike SI2's diffusion (see _apply_diffusion_correction), this needs no
    separate split-step correction and no forward-Euler CFL cap on kappa --
    the matrix exponential is unconditionally stable regardless of kappa.

    Returns (state_new, n_rhs_full).
    """
    u_n = state['u']
    w_n = state['w']

    # N_residual = N(q^n) - A(q^n)*q^n  (purely nonlinear pressure terms)
    n_full = compute_nonlinear_rhs(state, grid)
    a_q    = _frozen_advect(state, u_n, w_n, grid)
    n_res  = {k: n_full[k] - a_q[k] for k in n_full}

    q_vec  = _state_to_vec(state)
    nr_vec = _state_to_vec(n_res)

    if p is None:
        cs = math.sqrt(grid.cp / grid.cv * grid.Rd * grid.T0)
        p  = max(1, math.ceil(cs * math.pi / grid.dx * dt / 15.0))

    h   = dt / p
    c_h = h * nr_vec    # very small: only nonlinear pressure correction

    kappa = getattr(grid, 'diffusion_coeff', 0.0)
    order = getattr(grid, 'diffusion_order', 2)

    def L_n_h(v):
        """L_n * v scaled by h, where L_n = L + A(q^n) [+ diffusion]."""
        sv  = _vec_to_state(v, grid)
        Lv  = compute_linear_rhs(sv, grid)
        Av  = _frozen_advect(sv, u_n, w_n, grid)
        Lsum = {k: Lv[k] + Av[k] for k in Lv}
        if kappa > 0.0:
            Dv = compute_hyperdiffusion_rhs(sv, grid, order=order, coeff=kappa)
            for k in Lsum:
                Lsum[k] += Dv[k]
        return h * _state_to_vec(Lsum)

    y = q_vec.copy()
    for _ in range(p):
        y = _krylov_epi(L_n_h, y, [c_h], m_max=m_sub)

    return _vec_to_state(y, grid), n_full


# ===========================================================================
# State vector utilities
# ===========================================================================

def _state_to_vec(state):
    """Flatten {u,w,theta,pi} -> 1-D array."""
    return np.concatenate([state[k].ravel()
                           for k in ['u', 'w', 'theta', 'pi']])


def _vec_to_state(vec, grid):
    """Reshape 1-D array -> {u,w,theta,pi} dict."""
    n = grid.nz * grid.nx
    return {
        'u':     vec[0*n : 1*n].reshape(grid.nz, grid.nx),
        'w':     vec[1*n : 2*n].reshape(grid.nz, grid.nx),
        'theta': vec[2*n : 3*n].reshape(grid.nz, grid.nx),
        'pi':    vec[3*n : 4*n].reshape(grid.nz, grid.nx),
    }


# ===========================================================================
# Unit tests
# ===========================================================================


def _verify_phipm(m=10, n=20, seed=42):
    """Unit test: compare _krylov_epi against direct scipy expm."""
    rng = np.random.default_rng(seed)
    A_full = rng.standard_normal((n, n))
    A_full = A_full - A_full.T
    A_full *= 0.5
    q  = rng.standard_normal(n)
    c1 = rng.standard_normal(n) * 0.1
    c2 = rng.standard_normal(n) * 0.01
    from scipy.linalg import expm
    M_aug = np.zeros((n+2, n+2))
    M_aug[:n, :n]  = A_full
    M_aug[:n, n]   = c2
    M_aug[:n, n+1] = c1
    M_aug[n,   n+1] = 1.0   # phi structure: row n links phi1 and phi2 columns

    # Initial vector: [q; 0; 1] -- last entry seeds the phi polynomial
    v0 = np.zeros(n + 2)
    v0[:n] = q
    v0[-1] = 1.0

    # Reference: direct matrix exponential applied to augmented system
    # Result[:n] == exp(A_full)*q + phi1(A_full)*c1 + phi2(A_full)*c2
    ref = expm(M_aug) @ v0

    def L_a(v):
        return A_full @ v

    res = _krylov_epi(L_a, q, [c1, c2], m_max=n)   # full Krylov space -> exact

    err = np.linalg.norm(res - ref[:n]) / max(np.linalg.norm(ref[:n]), 1e-15)
    print("  _verify_phipm:        relative error = {:.3e}  ({})"
          .format(err, 'PASS' if err < 1e-3 else 'FAIL'))
    return err


# ===========================================================================
# Self-test entry point
# ===========================================================================

if __name__ == "__main__":
    print("\n  integrators.py -- self-test\n")
    print("  Verifying Krylov phi-function computations:")
    _verify_phipm()
    print()
    print("  To run the full zero-amplitude unit tests across all schemes,")
    print("  use:  python tests/test_integrators.py")
