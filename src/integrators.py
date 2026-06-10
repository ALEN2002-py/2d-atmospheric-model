"""
integrators.py
==============
All time integration schemes for the 2D atmospheric model.

FTCS   — Forward Euler            (explicit, 1st order)
BTCS   — Backward Euler           (implicit, 1st order, linearised)
CTCS   — Leapfrog                 (explicit, 2nd order)
RK4    — Classical Runge-Kutta    (explicit, 4th order)
SI     — Semi-implicit IMEX       (implicit L, explicit N, 2nd order)
EPI2   — Exponential Propagation  (exponential L, explicit N, 2nd order)
EPI3   — Exponential Propagation  (exponential L, explicit N, 3rd order)

Key concept — L + N splitting (from dynamics.py):
  dq/dt = L*q + N(q)
  L = linear stiff part (acoustic waves, buoyancy)
  N = nonlinear slow part (advection)

FTCS/BTCS/CTCS/RK4: treat full RHS together, no splitting.
SI:   treat L implicitly (removes acoustic CFL), N explicitly.
EPI2: compute e^(L*dt) exactly via Krylov subspace.
EPI3: EPI2 + correction term using 2nd difference of N.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.linalg import expm as small_expm
from dynamics import (compute_rhs, compute_linear_rhs,
                      compute_nonlinear_rhs)


# ===========================================================================
# Main dispatcher
# ===========================================================================

def step(state, grid, dt, scheme='RK4', state_old=None):
    """
    Advance model state by one time step dt.

    Parameters
    ----------
    state     : dict  — current state {u, w, theta, pi}
    grid      : Grid
    dt        : float — time step [s]
    scheme    : str   — 'FTCS','BTCS','CTCS','RK4','SI','EPI2','EPI3'
    state_old : dict  — previous state (CTCS only)

    Returns
    -------
    state_new : dict
    state_old : dict  — for CTCS bookkeeping
    """
    if scheme == 'FTCS':
        return _ftcs(state, grid, dt), state

    elif scheme == 'BTCS':
        return _btcs(state, grid, dt), state

    elif scheme == 'CTCS':
        if state_old is None:
            print("  CTCS: bootstrapping step 0 with FTCS")
            return _ftcs(state, grid, dt), state
        return _ctcs(state, state_old, grid, dt), state

    elif scheme == 'RK4':
        return _rk4(state, grid, dt), state

    elif scheme == 'SI':
        return _semi_implicit(state, grid, dt), state

    elif scheme == 'EPI2':
        return _epi2(state, grid, dt), state

    elif scheme == 'EPI3':
        return _epi3(state, grid, dt), state

    else:
        raise ValueError(f"Unknown scheme '{scheme}'.")


# ===========================================================================
# Scheme 1 — FTCS (Forward Euler)
# ===========================================================================

def _ftcs(state, grid, dt):
    """
    phi^(n+1) = phi^n + dt * F(phi^n)
    1st order. Unconditionally unstable for oscillatory problems.
    """
    rhs = compute_rhs(state, grid)
    return {k: state[k] + dt * rhs[k] for k in state}


# ===========================================================================
# Scheme 2 — BTCS (Backward Euler, one Picard iteration)
# ===========================================================================

def _btcs(state, grid, dt):
    """
    phi* = phi^n + dt * F(phi^n)         [predict]
    phi^(n+1) = phi^n + dt * F(phi*)     [correct]
    1st order. Gives implicit character via one Picard iteration.
    Full implicit solve handled by SI scheme.
    """
    rhs_n     = compute_rhs(state, grid)
    state_star = {k: state[k] + dt * rhs_n[k] for k in state}
    rhs_star  = compute_rhs(state_star, grid)
    return {k: state[k] + dt * rhs_star[k] for k in state}


# ===========================================================================
# Scheme 3 — CTCS (Leapfrog)
# ===========================================================================

def _ctcs(state, state_old, grid, dt):
    """
    phi^(n+1) = phi^(n-1) + 2*dt * F(phi^n)
    2nd order. Neutral amplitude |A|=1. Has computational mode.
    Use robert_asselin_filter() after each step to control the mode.
    """
    rhs = compute_rhs(state, grid)
    return {k: state_old[k] + 2 * dt * rhs[k] for k in state}


def robert_asselin_filter(state_old, state, state_new, alpha=0.1):
    """
    Robert-Asselin filter to damp leapfrog computational mode.
    Applied after each CTCS step:
      phi_filtered = phi^n + alpha*(phi^(n-1) - 2*phi^n + phi^(n+1))
    Note: reduces 2nd-order CTCS to 1st order.
    """
    return {
        k: state[k] + alpha * (state_old[k] - 2*state[k] + state_new[k])
        for k in state
    }


# ===========================================================================
# Scheme 4 — RK4 (Classical 4-stage Runge-Kutta)
# ===========================================================================

def _rk4(state, grid, dt):
    """
    k1 = F(phi^n)
    k2 = F(phi^n + dt/2 * k1)
    k3 = F(phi^n + dt/2 * k2)
    k4 = F(phi^n + dt   * k3)
    phi^(n+1) = phi^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    4th order. Max stable wDt ~ 2.82. Primary explicit benchmark scheme.
    """
    def add(s, scale, k):
        return {var: s[var] + scale * k[var] for var in s}

    k1 = compute_rhs(state, grid)
    k2 = compute_rhs(add(state, dt/2, k1), grid)
    k3 = compute_rhs(add(state, dt/2, k2), grid)
    k4 = compute_rhs(add(state, dt,   k3), grid)

    return {var: state[var] + (dt/6)*(k1[var] + 2*k2[var]
                                      + 2*k3[var] + k4[var])
            for var in state}


# ===========================================================================
# Scheme 5 — Semi-Implicit (IMEX Crank-Nicolson)
# ===========================================================================

def _semi_implicit(state, grid, dt):
    """
    IMEX Crank-Nicolson: treat L (acoustic/buoyancy) implicitly,
    N (advection) explicitly.

    Update formula:
      (I - dt/2 * L) q^(n+1) = (I + dt/2 * L) q^n + dt * N(q^n)

    Removes the acoustic CFL constraint — time step limited by
    advective speed (~10 m/s) rather than acoustic speed (~347 m/s).

    The linear system is solved iteratively using scipy GMRES.
    L is represented as a LinearOperator (no explicit matrix storage).
    """
    nz  = grid.nz
    nx  = grid.nx
    N   = nz * nx        # points per variable
    dim = 4 * N          # total degrees of freedom

    # Flatten state to 1D vector: [u_flat, w_flat, theta_flat, pi_flat]
    q_n = _state_to_vec(state)

    # Compute nonlinear RHS at current time (explicit)
    rhs_n = compute_nonlinear_rhs(state, grid)
    n_vec = _state_to_vec(rhs_n)

    # RHS of linear system: (I + dt/2 * L) q^n + dt * N(q^n)
    def L_apply(q_vec):
        """Apply linear operator L to a flat state vector."""
        s   = _vec_to_state(q_vec, grid)
        Ls  = compute_linear_rhs(s, grid)
        return _state_to_vec(Ls)

    # Build RHS: (I + dt/2 * L) q^n + dt * N(q^n)
    rhs_vec = q_n + (dt/2) * L_apply(q_n) + dt * n_vec

    # Build linear operator for LHS: A * q^(n+1) = rhs_vec
    # A = (I - dt/2 * L)
    def matvec(q_vec):
        return q_vec - (dt/2) * L_apply(q_vec)

    A = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)

    # Solve using GMRES
    q_new, info = gmres(A, rhs_vec, x0=q_n, rtol=1e-8, maxiter=200,
                        restart=50)
    if info != 0:
        print(f"  [SI] GMRES did not converge (info={info}). "
              f"Using initial guess.")
        q_new = q_n + dt * (L_apply(q_n) + n_vec)

    return _vec_to_state(q_new, grid)


# ===========================================================================
# Scheme 6 — EPI2 (Exponential Propagation Iterative, 2nd order)
# ===========================================================================

def _epi2(state, grid, dt):
    """
    EPI2: exactly propagates the linear part using the matrix exponential.

    q^(n+1) = exp(L*dt) * q^n + dt * phi1(L*dt) * N(q^n)

    where phi1(z) = (exp(z) - 1) / z

    Both exp(L*dt)*v and phi1(L*dt)*v are computed simultaneously
    using the Krylov subspace Arnoldi algorithm without forming L
    as an explicit matrix.

    Advantage over explicit schemes: no stability constraint on L.
    Time step limited only by accuracy of the nonlinear advection.
    """
    # Nonlinear RHS at current time
    n_rhs = compute_nonlinear_rhs(state, grid)
    n_vec = _state_to_vec(n_rhs)

    q_n   = _state_to_vec(state)

    # Apply L to a flat vector
    def L_apply(v):
        s  = _vec_to_state(v, grid)
        Ls = compute_linear_rhs(s, grid)
        return _state_to_vec(Ls)

    # Compute exp(L*dt)*q^n  and  phi1(L*dt)*N(q^n) via Krylov
    exp_q, phi1_n = _krylov_epi(L_apply, q_n, n_vec, dt, m=30)

    q_new = exp_q + dt * phi1_n
    return _vec_to_state(q_new, grid)


# ===========================================================================
# Scheme 7 — EPI3 (Exponential Propagation Iterative, 3rd order)
# ===========================================================================

def _epi3(state, grid, dt):
    """
    EPI3: EPI2 + correction term using difference of N.

    q^(n+1) = exp(L*dt)*q^n + dt*phi1(L*dt)*N(q^n)
              + dt^2 * phi2(L*dt) * (N(q*) - N(q^n)) / dt

    where:
      phi2(z) = (exp(z) - 1 - z) / z^2
      q* = EPI2 solution (intermediate)

    The correction term captures the variation of N during the step,
    giving 3rd-order accuracy compared to EPI2's 2nd order.
    """
    q_n   = _state_to_vec(state)

    # Nonlinear RHS at current time
    n_rhs = compute_nonlinear_rhs(state, grid)
    n_vec = _state_to_vec(n_rhs)

    def L_apply(v):
        s  = _vec_to_state(v, grid)
        Ls = compute_linear_rhs(s, grid)
        return _state_to_vec(Ls)

    # --- Step 1: EPI2 intermediate solution ---
    exp_q, phi1_n = _krylov_epi(L_apply, q_n, n_vec, dt, m=30)
    q_star_vec    = exp_q + dt * phi1_n
    state_star    = _vec_to_state(q_star_vec, grid)

    # --- Step 2: Nonlinear RHS at intermediate state ---
    n_star_rhs = compute_nonlinear_rhs(state_star, grid)
    n_star_vec = _state_to_vec(n_star_rhs)

    # --- Step 3: Correction using phi2 ---
    # d_N = (N(q*) - N(q^n)) / dt  — finite difference of N
    d_n_vec = (n_star_vec - n_vec) / dt

    # Compute phi2(L*dt) * d_N via Krylov
    _, phi2_d = _krylov_epi(L_apply, q_n, d_n_vec, dt, m=30,
                             return_phi2=True)

    q_new = q_star_vec + dt**2 * phi2_d
    return _vec_to_state(q_new, grid)


# ===========================================================================
# Krylov subspace helper — Arnoldi algorithm for phi functions
# ===========================================================================

def _krylov_epi(L_apply, q_vec, b_vec, dt, m=30, return_phi2=False):
    """
    Compute exp(L*dt)*q  and  phi1(L*dt)*b simultaneously
    using the Arnoldi iteration (Krylov subspace method).

    This avoids forming L as an explicit matrix. Instead we build
    an m-dimensional orthonormal basis V_m for the Krylov space
    span{q, L*q, L^2*q, ...} and compute phi functions on the
    small (m x m) Hessenberg matrix H_m.

    Algorithm (Hochbruck & Ostermann 2010):
    1. Build Arnoldi basis and H_m using q as starting vector
    2. Form augmented vector w = [q; b; 0; 1] and augmented H
    3. Compute exp of small augmented H using scipy.linalg.expm
    4. Project back to full space

    Parameters
    ----------
    L_apply     : callable  — applies L to a flat vector
    q_vec       : 1D array  — current state vector
    b_vec       : 1D array  — nonlinear RHS vector
    dt          : float     — time step
    m           : int       — Krylov subspace dimension
    return_phi2 : bool      — also return phi2(L*dt)*b

    Returns
    -------
    exp_q  : exp(L*dt) * q_vec
    phi1_b : phi1(L*dt) * b_vec
    phi2_b : phi2(L*dt) * b_vec  (only if return_phi2=True)
    """
    n  = len(q_vec)
    m  = min(m, n - 2)

    # --- Arnoldi iteration: build Krylov basis for L ---
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    beta = np.linalg.norm(q_vec)
    if beta < 1e-15:
        zeros = np.zeros(n)
        if return_phi2:
            return zeros, phi1_direct(L_apply, b_vec, dt, n), \
                   phi2_direct(L_apply, b_vec, dt, n)
        return zeros, phi1_direct(L_apply, b_vec, dt, n)

    V[:, 0] = q_vec / beta

    j_max = 0
    for j in range(m):
        w = L_apply(V[:, j]) * dt

        # Modified Gram-Schmidt orthogonalisation
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w        = w - H[i, j] * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)
        j_max = j + 1
        if H[j + 1, j] < 1e-12:
            break                   # Krylov space exhausted
        V[:, j + 1] = w / H[j + 1, j]

    m_eff = j_max

    # --- Augmented system to compute exp and phi1 simultaneously ---
    # Augment the Hessenberg matrix with columns for b_vec
    # Size: (m_eff + 2) x (m_eff + 2)
    H_aug = np.zeros((m_eff + 2, m_eff + 2))
    H_aug[:m_eff, :m_eff] = H[:m_eff, :m_eff]

    # b projected onto Krylov basis: V^T * b_vec
    b_proj = V[:, :m_eff].T @ b_vec
    H_aug[:m_eff, m_eff] = b_proj

    # Last diagonal entry for phi1 scaling
    H_aug[m_eff, m_eff + 1] = 1.0

    if return_phi2:
        H_aug2 = np.zeros((m_eff + 3, m_eff + 3))
        H_aug2[:m_eff + 2, :m_eff + 2] = H_aug
        H_aug2[m_eff + 1, m_eff + 2]   = 1.0
        exp_H2 = small_expm(H_aug2)
        # Extract components
        e1     = np.zeros(m_eff + 3)
        e1[0]  = 1.0
        y      = exp_H2 @ e1
        exp_q  = beta * (V[:, :m_eff] @ y[:m_eff])
        phi1_b = V[:, :m_eff] @ exp_H2[:m_eff, m_eff]
        phi2_b = V[:, :m_eff] @ exp_H2[:m_eff, m_eff + 1]
        # Handle residual from b not fully in Krylov space
        b_perp = b_vec - V[:, :m_eff] @ b_proj
        if np.linalg.norm(b_perp) > 1e-10:
            phi1_b += _phi1_matvec(L_apply, b_perp, dt, n)
        return exp_q, phi1_b, phi2_b

    # Standard EPI2: just exp and phi1
    exp_H = small_expm(H_aug)
    e1    = np.zeros(m_eff + 2)
    e1[0] = 1.0
    y     = exp_H @ e1

    exp_q  = beta * (V[:, :m_eff] @ y[:m_eff])
    phi1_b = V[:, :m_eff] @ exp_H[:m_eff, m_eff]

    # Handle component of b_vec orthogonal to Krylov space
    b_perp = b_vec - V[:, :m_eff] @ b_proj
    if np.linalg.norm(b_perp) > 1e-10:
        phi1_b += _phi1_matvec(L_apply, b_perp, dt, n)

    return exp_q, phi1_b


def _phi1_matvec(L_apply, v, dt, n, m=20):
    """
    Compute phi1(L*dt)*v using separate Krylov iteration.
    phi1(z) = (exp(z) - 1) / z
    Used for the component of b orthogonal to main Krylov space.
    """
    m  = min(m, n - 1)
    V  = np.zeros((n, m + 1))
    H  = np.zeros((m + 1, m))

    beta = np.linalg.norm(v)
    if beta < 1e-15:
        return np.zeros(n)

    V[:, 0] = v / beta
    j_max   = 0

    for j in range(m):
        w = L_apply(V[:, j]) * dt
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w        = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        j_max = j + 1
        if H[j + 1, j] < 1e-12:
            break
        V[:, j + 1] = w / H[j + 1, j]

    m_eff = j_max
    Hm    = H[:m_eff, :m_eff]
    expHm = small_expm(Hm)

    # phi1(H) = (exp(H) - I) @ inv(H)  computed numerically
    try:
        from numpy.linalg import solve
        phi1_Hm = solve(Hm.T, (expHm - np.eye(m_eff)).T).T
    except np.linalg.LinAlgError:
        phi1_Hm = expHm @ np.eye(m_eff)   # fallback

    e1    = np.zeros(m_eff)
    e1[0] = 1.0
    return beta * (V[:, :m_eff] @ (phi1_Hm @ e1))


def phi1_direct(L_apply, v, dt, n, m=20):
    """Compute phi1(L*dt)*v directly."""
    return _phi1_matvec(L_apply, v, dt, n, m)


def phi2_direct(L_apply, v, dt, n, m=20):
    """Compute phi2(L*dt)*v. phi2(z) = (exp(z) - 1 - z) / z^2"""
    m  = min(m, n - 1)
    V  = np.zeros((n, m + 1))
    H  = np.zeros((m + 1, m))
    beta = np.linalg.norm(v)
    if beta < 1e-15:
        return np.zeros(n)
    V[:, 0] = v / beta
    j_max   = 0
    for j in range(m):
        w = L_apply(V[:, j]) * dt
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w        = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        j_max = j + 1
        if H[j + 1, j] < 1e-12:
            break
        V[:, j + 1] = w / H[j + 1, j]
    m_eff = j_max
    Hm    = H[:m_eff, :m_eff]
    expHm = small_expm(Hm)
    Im    = np.eye(m_eff)
    try:
        from numpy.linalg import solve
        phi2_Hm = solve(Hm.T, (solve(Hm.T, (expHm - Im).T).T - Im).T).T
    except np.linalg.LinAlgError:
        phi2_Hm = Im
    e1    = np.zeros(m_eff)
    e1[0] = 1.0
    return beta * (V[:, :m_eff] @ (phi2_Hm @ e1))


# ===========================================================================
# State vector utilities
# ===========================================================================

def _state_to_vec(state):
    """Flatten state dict to 1D numpy array: [u, w, theta, pi]."""
    return np.concatenate([state[k].ravel()
                           for k in ['u', 'w', 'theta', 'pi']])


def _vec_to_state(vec, grid):
    """Reshape 1D vector back to state dict."""
    N  = grid.nz * grid.nx
    return {
        'u':     vec[0*N : 1*N].reshape(grid.nz, grid.nx),
        'w':     vec[1*N : 2*N].reshape(grid.nz, grid.nx),
        'theta': vec[2*N : 3*N].reshape(grid.nz, grid.nx),
        'pi':    vec[3*N : 4*N].reshape(grid.nz, grid.nx),
    }


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from grid import Grid

    g  = Grid()
    dt = 0.02

    print("Zero amplitude test — all schemes (max change should be ~0):\n")
    for scheme in ['FTCS', 'BTCS', 'CTCS', 'RK4', 'SI', 'EPI2', 'EPI3']:
        state     = g.allocate_state()
        state_old = g.allocate_state()
        try:
            s_new, _ = step(state, g, dt, scheme=scheme,
                            state_old=state_old)
            mc = max(np.max(np.abs(s_new[k])) for k in s_new)
            print(f"  {scheme:<6}  max change = {mc:.2e}  "
                  f"{'PASS' if mc < 1e-10 else 'FAIL'}")
        except Exception as e:
            print(f"  {scheme:<6}  ERROR: {e}")
