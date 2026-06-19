"""
integrators.py
==============
All time integration schemes for the 2D atmospheric model.

FTCS   — Forward Euler            (explicit, 1st order)
BTCS   — Heun's method            (explicit, 2nd order; mislabelled, not backward Euler)
CTCS   — Leapfrog                 (explicit, 2nd order)
RK4    — Classical Runge-Kutta    (explicit, 4th order)
SI     — Semi-implicit IMEX       (1st order, L implicit via GMRES)
EPI2   — Exponential Propagation  (2nd order, Krylov sub-steps)
EPI3   — Exponential Propagation  (3rd order, P&C 2022 formula)

L + N splitting:
  dq/dt = L*q + N(q)
  L = linear stiff operator (acoustic waves, buoyancy)
  N = nonlinear advection (slow, treated explicitly)

EPI formulas (Pudykiewicz & Clancy 2022, eqs 2.6-2.7)
-------------------------------------------------------
EPI2:  u^{n+1} = u^n + phi_1(Jn*dt)*dt*F^n                      (eq 2.6)
EPI3:  u^{n+1} = u^n + phi_1(Jn*dt)*dt*F^n
                      + (2/3)*phi_2(Jn*dt)*dt*R^{n-1}            (eq 2.7)
       where R^{n-1} = F^{n-1} - F^n - Jn*(u^{n-1} - u^n)

With the approximation Jn = L (constant linear part):
  F^n = L*q^n + N^n
  phi_1(L*dt)*dt*F^n = (exp(L*dt)-I)*q^n + phi_1(L*dt)*dt*N^n
  -> EPI2 == exp(L*dt)*q^n + phi_1(L*dt)*dt*N^n   (same as before)
  R^{n-1} = N^{n-1} - N^n

So EPI3 correction = (2/3)*phi_2(L*dt)*dt*(N^{n-1} - N^n)
This uses the PREVIOUS step's N, unlike the predictor-corrector.

Shapiro filter (P&C eq 5.6-5.7)
---------------------------------
Applied to all 4 fields every 2 EPI time steps.
F = Fx*Fz (separable box filter):
  Fx: 1/4 * f_{i-1} + 1/2 * f_i + 1/4 * f_{i+1}   (periodic in x)
  Fz: 1/4 * f_{j-1} + 1/2 * f_j + 1/4 * f_{j+1}   (zero-gradient at top/bottom)

EPI sub-step approach for EPI2
--------------------------------
For large acoustic CFL, single Krylov m=30 can't span exp(L*dt).
Sub-divide into p sub-steps of h=dt/p (auto-selected so c_s*pi/dx*h <= 15).
EPI2 exact identity: exp(L*dt)*q + phi_1(L*dt)*dt*N
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
  - epi_extra  : dict {'n_rhs': n_rhs} for EPI2/EPI3; None otherwise
Callers that used the old 2-tuple can ignore the third element.
"""

import math
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.linalg import expm as small_expm
from dynamics import (compute_rhs, compute_linear_rhs,
                      compute_nonlinear_rhs)


# ===========================================================================
# Main dispatcher
# ===========================================================================

def step(state, grid, dt, scheme='RK4', state_old=None, epi_n_prev=None):
    """
    Advance model state by one time step dt.

    Parameters
    ----------
    state      : dict  — current state {u, w, theta, pi}
    grid       : Grid
    dt         : float — time step [s]
    scheme     : str   — 'FTCS','BTCS','CTCS','RK4','SI','EPI2','EPI3'
    state_old  : dict  — previous state (CTCS only, pass None for auto-bootstrap)
    epi_n_prev : dict  — previous nonlinear RHS (EPI3 only; None on first step)

    Returns
    -------
    state_new  : dict
    state_prev : dict  (= input state, for CTCS bookkeeping)
    epi_extra  : dict  {'n_rhs': ...} for EPI2/EPI3, else None
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
        return _semi_implicit(state, grid, dt), state, None
    elif scheme == 'EPI2':
        state_new, n_rhs = _epi2(state, grid, dt)
        return state_new, state, {'n_rhs': n_rhs}
    elif scheme == 'EPI3':
        state_new, n_rhs = _epi3(state, grid, dt, n_prev=epi_n_prev)
        return state_new, state, {'n_rhs': n_rhs}
    else:
        raise ValueError(f"Unknown scheme '{scheme}'.")


# ===========================================================================
# Shapiro filter (Pudykiewicz & Clancy 2022, eq 5.6-5.7)
# ===========================================================================

def shapiro_filter(state, grid):
    """
    2D Shapiro (box) filter applied to all 4 state fields.

    F = Fx * Fz  (applied sequentially, both separable):
      Fx: f_i -> 1/4*f_{i-1} + 1/2*f_i + 1/4*f_{i+1}   (periodic in x)
      Fz: f_j -> 1/4*f_{j-1} + 1/2*f_j + 1/4*f_{j+1}   (zero-gradient BCs at top/bottom)

    Applied every 2 EPI time steps (call from run loop when (step+1) % 2 == 0).
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

    q_new, info = gmres(A, q_rhs, x0=q0, atol=1e-10, rtol=1e-8)

    if info != 0:
        print(f"  SI: GMRES did not converge (info={info})")

    return _vec_to_state(q_new, grid)


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
    if beta < 1e-15:
        return np.zeros(n)

    m = min(m_max, n)
    V = np.zeros((m + 1, n))
    H = np.zeros((m + 1, m))
    V[0] = q_vec / beta
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
# Scheme 6 — EPI2  (sub-step Krylov, P&C 2022 eq 2.6)
# ===========================================================================

def _epi2(state, grid, dt, p=None, m_sub=10):
    """
    EPI2 (P&C 2022 eq 2.6):
      u^{n+1} = u^n + phi_1(L*dt)*dt*F^n
             == exp(L*dt)*q^n + phi_1(L*dt)*dt*N^n   (with J=L approximation)

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

    Bootstrap: n_prev=None on the first step falls back to EPI2 (c2=0).

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
        # Bootstrap with EPI2 (no R_prev available yet)
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
        # Correction would dominate — use EPI2 for this step instead.
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
# =======


def _verify_phi2_krylov(m=15, n=20, seed=42):
    """
    Unit test: compare _krylov_phi2 against direct scipy expm formula.
    phi_2(A)*b = expm([[A, b, 0]; [0, 0, 1]; [0, 0, 0]]) @ [0; 0; 1]  first n entries.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.5
    A = A - A.T   # skew-symmetric (pure imaginary eigenvalues)
    b = rng.standard_normal(n) * 0.1

    from scipy.linalg import expm
    # Direct reference: phi_2(A)*b via augmented matrix [[A, b, 0]; [0, 0, 1]; [0, 0, 0]]
    aug = n + 2
    Ms2 = np.zeros((aug, aug))
    Ms2[:n, :n] = A
    Ms2[:n, n]  = b
    Ms2[n, n+1] = 1.0
    v02 = np.zeros(aug); v02[-1] = 1.0
    ref_b = expm(Ms2) @ v02   # first n = phi_2(A)*b

    def L_a(v):
        return A @ v

    res = _krylov_phi2(L_a, b, m_max=n)   # full space -> exact

    err = np.linalg.norm(res - ref_b[:n]) / max(np.linalg.norm(ref_b[:n]), 1e-15)
    print("  _verify_phi2_krylov: relative error = %.3e  (%s)"
          % (err, 'PASS' if err < 1e-3 else 'FAIL'))
    return err


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
    M_aug[n,  n+1] = 1.0
    v0 = np.zeros(n+2); v0[:n] = q; v0[-1] = 1.0
    ref = expm(M_aug) @ v0
    def L_apply(v):
        return A_full @ v
    res = _krylov_epi(L_apply, q, [c1, c2], m_max=m)
    err = np.linalg.norm(res - ref[:n]) / max(np.linalg.norm(ref[:n]), 1e-15)
    print("  _verify_phipm: relative error = %.3e  (%s)" % (err, 'PASS' if err < 1e-3 else 'FAIL'))
    return err


if __name__ == "__main__":
    _verify_phipm()
    _verify_phi2_krylov()
