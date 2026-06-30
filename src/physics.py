"""
physics.py
==========
Physical constants for the 2D non-hydrostatic atmospheric model.

All values are consistent with Giraldo & Restelli (2008) and
Pudykiewicz & Clancy (2022).

These are provided as a convenient reference module. The Grid class
also stores most of these as instance attributes (grid.g, grid.cp, etc.)
so experiments do not need to import this module directly.

Constants
---------
g   = 9.81 m/s^2         gravitational acceleration
cp  = 1004.0 J/kg/K      specific heat at constant pressure
cv  = 717.0 J/kg/K       specific heat at constant volume
Rd  = 287.0 J/kg/K       specific gas constant for dry air
p0  = 1.0e5 Pa           reference surface pressure
T0  = 300.0 K            reference temperature (= theta_bar for isentropic base state)

Derived
-------
gamma = cp/cv ~ 1.4       ratio of specific heats
kappa = Rd/cp ~ 0.2857    Poisson exponent (used in Exner pressure definition)
c_s   ~ 347 m/s           speed of sound at T0=300K  (= sqrt(gamma*Rd*T0))
N_iso ~ 0.0179 s^-1       Brunt-Vaisala frequency for isothermal base state
       (period ~ 351 s; causes bubble oscillation instead of rise)
"""

# ---------------------------------------------------------------------------
# Thermodynamic constants
# ---------------------------------------------------------------------------

G_GRAV = 9.81       # gravitational acceleration          [m/s^2]
CP     = 1004.0     # specific heat at constant pressure  [J/kg/K]
CV     = 717.0      # specific heat at constant volume    [J/kg/K]
RD     = 287.0      # specific gas constant for dry air   [J/kg/K]
P0     = 1.0e5      # reference pressure                  [Pa]
T0     = 300.0      # reference / base-state temperature  [K]

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

GAMMA   = CP / CV                           # ratio of specific heats        ~ 1.4
KAPPA   = RD / CP                           # Poisson exponent               ~ 0.2857
C_SOUND = (GAMMA * RD * T0) ** 0.5         # speed of sound at T0=300K      ~ 347 m/s

# Brunt-Vaisala frequency for an isothermal base state (T = T0 = const):
#   N^2 = g^2 / (cp * T0)  ~ 3.2e-4 s^-2
#   N   ~ 0.0179 s^-1,  period ~ 351 s
# This is why the isothermal base state causes the bubble to oscillate
# rather than rise freely (as it does in the isentropic base state).
N_ISOTHERMAL = (G_GRAV**2 / (CP * T0)) ** 0.5   # [s^-1]
