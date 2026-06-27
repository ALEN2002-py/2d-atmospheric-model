"""
grid.py
=======
Unstaggered 2D grid for the non-hydrostatic compressible atmospheric model.

All four prognostic variables (u, w, theta', pi') share the same set of
cell-centred grid points. A staggered Arakawa C-grid was considered but
an unstaggered layout was adopted because enforcing the solid-wall boundary
conditions cleanly is more important at this stage than the accuracy gains
from staggering.

Variables follow the theta-pi perturbation form of Giraldo & Restelli (2008):
  u      — horizontal velocity      [m/s]
  w      — vertical velocity        [m/s]
  theta' — potential temperature perturbation  [K]
  pi'    — Exner pressure perturbation         [-]

Grid layout  (nz rows x nx columns, cell centres):
  x_i = (i + 0.5) * dx,   i = 0 .. nx-1
  z_k = (k + 0.5) * dz,   k = 0 .. nz-1

Boundary conditions:
  x — periodic
  z — w = 0 at top and bottom (solid walls);
      one-sided finite differences at k=0 and k=nz-1
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Domain
    "Lx": 1000.0,   # m
    "Lz": 1000.0,   # m

    # Resolution
    "dx": 10.0,     # m
    "dz": 10.0,     # m

    # Physical constants
    "g":   9.81,    # m/s^2
    "cp":  1004.0,  # J/kg/K
    "cv":  717.0,   # J/kg/K
    "Rd":  287.0,   # J/kg/K
    "p0":  1.0e5,   # Pa

    # Base state
    "T0":             300.0,         # K  reference temperature / constant theta_bar
    "stratification": "isentropic",  # "isentropic" neutral dtheta/dz=0  (G&R 2008)
                                     # "isothermal"  stable, constant T0

    # Explicit diffusion coefficient.  0.0 = no diffusion.
    # Units: m^diffusion_order / s
    # G&R (2008) use κ ≈ 75 m²/s (order=2) for density current only.
    # For bubble tests use κ ≤ 5 m²/s or order=4/8 hyperdiffusion.
    "diffusion_coeff": 0.0,

    # Hyperdiffusion order: 2 = standard Laplacian (κ∇²),
    # 4 = biharmonic (-κ∇⁴), 8 = octaharmonic (-κ∇⁸).
    # Higher order damps only the shortest waves, preserving the bubble.
    "diffusion_order": 2,

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
        g = Grid()                                        # isentropic default
        g = Grid({"Lx": 5000, "dx": 20})                 # custom domain
        g = Grid({"stratification": "isothermal"})        # stable base state
    """

    def __init__(self, params: dict = None):

        cfg = {**DEFAULTS, **(params or {})}

        # ------------------------------------------------------------------
        # 1. Grid spacing and cell count
        # ------------------------------------------------------------------
        self.Lx = cfg["Lx"]
        self.Lz = cfg["Lz"]
        self.dx = cfg["dx"]
        self.dz = cfg["dz"]

        self.nx = int(round(self.Lx / self.dx))
        self.nz = int(round(self.Lz / self.dz))

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

        self.kappa = self.Rd / self.cp   # R/cp ~ 0.2857

        # ------------------------------------------------------------------
        # 3. Coordinate arrays (cell centres)
        # ------------------------------------------------------------------
        self.x_1d = (np.arange(self.nx) + 0.5) * self.dx
        self.z_1d = (np.arange(self.nz) + 0.5) * self.dz
        self.x_2d, self.z_2d = np.meshgrid(self.x_1d, self.z_1d)

        # ------------------------------------------------------------------
        # 4. Base state profiles (function of z only)
        # ------------------------------------------------------------------
        # "isentropic" (DEFAULT): neutral stratification, dtheta_bar/dz = 0.
        #   Matches G&R (2008) Case 2. Bubble rises freely to top of domain.
        # "isothermal": stable stratification, theta_bar increases with height.
        #   Brunt-Vaisala period ~350 s -> bubble oscillates, does NOT rise.
        #   Wrong for rising-bubble benchmarks; kept for reference only.
        self._stratification  = cfg.get("stratification", "isentropic")
        self.diffusion_coeff  = float(cfg.get("diffusion_coeff", 0.0))
        self.diffusion_order  = int(cfg.get("diffusion_order", 2))

        self.pi_bar, self.theta_bar, self.dtheta_bar_dz = self._build_base_state()

        # ------------------------------------------------------------------
        # 5. Sponge layer
        # ------------------------------------------------------------------
        self.sponge = self._build_sponge(
            frac     = cfg["sponge_fraction"],
            strength = cfg["sponge_strength"],
        )

    # -----------------------------------------------------------------------
    # Private: base state
    # -----------------------------------------------------------------------

    def _build_base_state(self):
        """
        Compute hydrostatic base state profiles.

        "isentropic"  (DEFAULT)
          Neutral stratification: theta_bar = T0 = constant.
          G&R (2008), most NWP benchmarks.
            theta_bar = T0  (everywhere)
            pi_bar    = 1 - g*z/(cp*T0)   (linear)
            dtheta_bar_dz = 0  -> no restoring force -> bubble rises freely.

        "isothermal"
          Stable stratification: temperature T0 = constant.
          theta_bar increases with height -> bubble oscillates instead of rising.
            p_bar     = p0 * exp(-g*z/(Rd*T0))
            pi_bar    = (p_bar/p0)^kappa
            theta_bar = T0 / pi_bar   (increases ~0.01 K/m for 1 km domain)
        """
        z     = self.z_1d
        strat = self._stratification

        if strat == "isentropic":
            # Neutral base state: constant theta_bar, linear pi_bar
            theta_bar     = np.full(self.nz, self.T0)
            pi_bar        = 1.0 - self.g * z / (self.cp * self.T0)
            dtheta_bar_dz = np.zeros(self.nz)

        else:
            # Stable (isothermal) base state
            p_bar     = self.p0 * np.exp(-self.g * z / (self.Rd * self.T0))
            pi_bar    = (p_bar / self.p0) ** self.kappa
            theta_bar = self.T0 / pi_bar

            dtheta_bar_dz        = np.zeros(self.nz)
            dtheta_bar_dz[1:-1]  = (theta_bar[2:] - theta_bar[:-2]) / (2*self.dz)
            dtheta_bar_dz[0]     = (theta_bar[1]  - theta_bar[0])   / self.dz
            dtheta_bar_dz[-1]    = (theta_bar[-1] - theta_bar[-2])  / self.dz

        return pi_bar, theta_bar, dtheta_bar_dz

    # -----------------------------------------------------------------------
    # Private: sponge
    # -----------------------------------------------------------------------

    def _build_sponge(self, frac, strength):
        """Rayleigh sponge in the top frac fraction of the domain."""
        z_s   = (1.0 - frac) * self.Lz
        alpha = np.zeros(self.nz)
        mask  = self.z_1d >= z_s
        if np.any(mask):
            zrel        = (self.z_1d[mask] - z_s) / (self.Lz - z_s)
            alpha[mask] = strength * np.sin(0.5 * np.pi * zrel) ** 2
        return alpha

    # -----------------------------------------------------------------------
    # Public: allocate state
    # -----------------------------------------------------------------------

    def allocate_state(self):
        """Return zero-filled state dict {u, w, theta, pi}, all shape (nz, nx)."""
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
        print(f"  Var shape  : {(self.nz, self.nx)}")
        print(f"  Strat.     : {self._stratification}")
        print(f"  theta_bar  : {self.theta_bar.min():.2f} to {self.theta_bar.max():.2f} K"
              f"  (dtheta/dz_max={self.dtheta_bar_dz.max():.5f} K/m)")
        print(f"  pi_bar     : {self.pi_bar.min():.4f} to {self.pi_bar.max():.4f}")
        print(f"  Sponge     : top {int(self.sponge.__class__.__name__ and 0.2*100)}%"
              f"  alpha_max={self.sponge.max():.4f}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ISENTROPIC (default) ===")
    g = Grid()
    g.info()

    print()
    print("=== ISOTHERMAL (old) ===")
    g2 = Grid({"stratification": "isothermal"})
    g2.info()
