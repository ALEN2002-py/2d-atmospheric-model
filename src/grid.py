"""
grid.py
=======
Unstaggered grid for the 2D non-hydrostatic atmospheric model.

IMPORTANT CHANGE FROM PREVIOUS VERSION
---------------------------------------
We switched from a staggered Arakawa C-grid to an UNSTAGGERED grid.

Why? Dr Clancy's note says:
  "we returned to unstaggered, finding that control of the boundary
   conditions was the more important factor"

On an unstaggered grid ALL variables live at the SAME points:
  u, w, theta', pi'  all at every (x_i, z_k) point.

This makes boundary conditions much simpler to apply and is the
starting point before we add complexity.

Variables used (theta-pi form, from Giraldo & Restelli 2008):
  u      : horizontal velocity                          [m/s]
  w      : vertical velocity                            [m/s]
  theta' : potential temperature perturbation           [K]
  pi'    : Exner pressure perturbation                  [-]

Grid layout (unstaggered):
--------------------------
  All variables on same (nz x nx) grid.
  Points at cell centres: x_i = (i+0.5)*dx, z_k = (k+0.5)*dz

  z ^
  nz|  *  *  *  *  *  *    <- all variables here
    |  *  *  *  *  *  *
    |  *  *  *  *  *  *
   0|  *  *  *  *  *  *
    +-------------------> x
     0                nx

  * = u, w, theta', pi' all live at same point

Boundary conditions:
  x: periodic   (left wraps to right)
  z: w=0 at top and bottom (solid walls)
     one-sided differences at top/bottom for z-derivatives
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default parameters — warm bubble (Robert 1993)
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Domain
    "Lx": 1000.0,   # m — 1 km wide  (Dr Clancy's note uses ~1km domain)
    "Lz": 1000.0,   # m — 1 km tall

    # Resolution
    "dx": 10.0,     # m — 10 m spacing
    "dz": 10.0,     # m — 10 m spacing

    # Physical constants
    "g":   9.81,    # m/s^2
    "cp":  1004.0,  # J/kg/K
    "cv":  717.0,   # J/kg/K
    "Rd":  287.0,   # J/kg/K
    "p0":  1.0e5,   # Pa  — reference pressure for Exner

    # Base state (isothermal)
    "T0":    300.0,   # K  — constant background temperature

    # Sponge layer
    "sponge_fraction": 0.20,
    "sponge_strength": 0.01,   # s^-1
}


# ---------------------------------------------------------------------------
# Grid class
# ---------------------------------------------------------------------------

class Grid:
    """
    Unstaggered 2D grid. All variables on the same set of points.

    Usage
    -----
        g = Grid()                            # defaults
        g = Grid({"Lx": 5000, "dx": 20})     # warm bubble
    """

    def __init__(self, params: dict = None):

        # Merge user params with defaults
        cfg = {**DEFAULTS, **(params or {})}

        # ------------------------------------------------------------------
        # 1. Grid spacing and cell count
        # ------------------------------------------------------------------
        self.Lx = cfg["Lx"]
        self.Lz = cfg["Lz"]
        self.dx = cfg["dx"]
        self.dz = cfg["dz"]

        self.nx = int(round(self.Lx / self.dx))   # number of cells in x
        self.nz = int(round(self.Lz / self.dz))   # number of cells in z

        assert np.isclose(self.nx * self.dx, self.Lx), "Lx not divisible by dx"
        assert np.isclose(self.nz * self.dz, self.Lz), "Lz not divisible by dz"

        # ------------------------------------------------------------------
        # 2. Physical constants
        # ------------------------------------------------------------------
        self.g   = cfg["g"]
        self.cp  = cfg["cp"]
        self.cv  = cfg["cv"]
        self.Rd  = cfg["Rd"]
        self.p0  = cfg["p0"]
        self.T0  = cfg["T0"]

        # Derived
        self.kappa = self.Rd / self.cp   # R/cp ~ 0.2857

        # ------------------------------------------------------------------
        # 3. Coordinate arrays (cell centres)
        # ------------------------------------------------------------------
        # x: cell centres at (i + 0.5)*dx for i = 0,...,nx-1
        self.x_1d = (np.arange(self.nx) + 0.5) * self.dx   # shape (nx,)

        # z: cell centres at (k + 0.5)*dz for k = 0,...,nz-1
        self.z_1d = (np.arange(self.nz) + 0.5) * self.dz   # shape (nz,)

        # 2D coordinate grids — useful for setting initial conditions
        # x_2d[k, i] = x-coordinate of point (i, k)
        # z_2d[k, i] = z-coordinate of point (i, k)
        self.x_2d, self.z_2d = np.meshgrid(self.x_1d, self.z_1d)
        # both shape: (nz, nx)

        # ------------------------------------------------------------------
        # 4. Base state profiles (function of z only)
        # ------------------------------------------------------------------
        # Isothermal base state: constant temperature T0 everywhere.
        # This satisfies hydrostatic balance:  d(pi_bar)/dz = -g/(cp*theta_bar)
        #
        # From this, we can show:
        #   pi_bar(z) = (p_bar/p0)^(R/cp)
        #   p_bar(z)  = p0 * exp(-g*z / (Rd*T0))   [exponential decay]
        #   theta_bar = T0 / pi_bar                  [definition of theta]
        #   d(theta_bar)/dz  (needed in equation 4)

        self.pi_bar, self.theta_bar, self.dtheta_bar_dz = self._build_base_state()
        # all shape: (nz,)

        # ------------------------------------------------------------------
        # 5. Sponge layer
        # ------------------------------------------------------------------
        self.sponge = self._build_sponge(
            frac     = cfg["sponge_fraction"],
            strength = cfg["sponge_strength"],
        )
        # shape: (nz,)

    # -----------------------------------------------------------------------
    # Private: base state
    # -----------------------------------------------------------------------

    def _build_base_state(self):
        """
        Compute isothermal hydrostatic base state profiles.

        Equations:
          p_bar(z) = p0 * exp(-g*z / (Rd*T0))   [from hydrostatic + ideal gas]
          pi_bar   = (p_bar/p0)^kappa             [definition of Exner pressure]
          theta_bar = T0 / pi_bar                 [definition of potential temp]

        Also computes d(theta_bar)/dz which appears in equation (4):
          ∂θ'/∂t = ... - w * d(θ_bar)/dz
        """
        z = self.z_1d   # z at cell centres, shape (nz,)

        # Background pressure (hydrostatic + isothermal)
        p_bar = self.p0 * np.exp(-self.g * z / (self.Rd * self.T0))

        # Exner pressure
        pi_bar = (p_bar / self.p0) ** self.kappa

        # Potential temperature
        theta_bar = self.T0 / pi_bar

        # Vertical gradient of theta_bar (needed for equation 4)
        # Use centred differences in interior, one-sided at boundaries
        dtheta_bar_dz = np.zeros(self.nz)
        # Interior: centred
        dtheta_bar_dz[1:-1] = (theta_bar[2:] - theta_bar[:-2]) / (2 * self.dz)
        # Bottom boundary: forward difference
        dtheta_bar_dz[0]    = (theta_bar[1] - theta_bar[0]) / self.dz
        # Top boundary: backward difference
        dtheta_bar_dz[-1]   = (theta_bar[-1] - theta_bar[-2]) / self.dz

        return pi_bar, theta_bar, dtheta_bar_dz

    # -----------------------------------------------------------------------
    # Private: sponge
    # -----------------------------------------------------------------------

    def _build_sponge(self, frac, strength):
        """
        Rayleigh sponge in the top `frac` fraction of the domain.
        alpha(z) ramps smoothly from 0 to `strength` using sin^2.
        """
        z_s   = (1.0 - frac) * self.Lz
        alpha = np.zeros(self.nz)
        mask  = self.z_1d >= z_s
        if np.any(mask):
            zrel         = (self.z_1d[mask] - z_s) / (self.Lz - z_s)
            alpha[mask]  = strength * np.sin(0.5 * np.pi * zrel) ** 2
        return alpha

    # -----------------------------------------------------------------------
    # Public: allocate state
    # -----------------------------------------------------------------------

    def allocate_state(self):
        """
        Return zero-filled arrays for all 4 prognostic variables.

        All variables have shape (nz, nx) — same grid, unstaggered.

        Returns
        -------
        dict:
            'u'     : (nz, nx)  horizontal velocity           [m/s]
            'w'     : (nz, nx)  vertical velocity             [m/s]
            'theta' : (nz, nx)  potential temp. perturbation  [K]
            'pi'    : (nz, nx)  Exner pressure perturbation   [-]
        """
        shape = (self.nz, self.nx)
        return {
            "u":     np.zeros(shape),
            "w":     np.zeros(shape),
            "theta": np.zeros(shape),
            "pi":    np.zeros(shape),
        }

    # -----------------------------------------------------------------------
    # Public: info
    # -----------------------------------------------------------------------

    def info(self):
        """Print grid summary."""
        print("=" * 50)
        print("Grid Summary (UNSTAGGERED)")
        print("=" * 50)
        print(f"  Domain     : {self.Lx} m x {self.Lz} m")
        print(f"  Spacing    : dx={self.dx} m,  dz={self.dz} m")
        print(f"  Cells      : nx={self.nx},  nz={self.nz}")
        print(f"  Var shape  : {(self.nz, self.nx)}  (all variables same)")
        print(f"  theta_bar  : {self.theta_bar.min():.2f} to {self.theta_bar.max():.2f} K")
        print(f"  pi_bar     : {self.pi_bar.min():.4f} to {self.pi_bar.max():.4f}")
        print(f"  Sponge     : top {int(0.2*100)}%  alpha_max={self.sponge.max():.4f}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    g = Grid()
    g.info()
    state = g.allocate_state()
    print("\nAll variable shapes:")
    for k, v in state.items():
        print(f"  {k:6s}: {v.shape}")
