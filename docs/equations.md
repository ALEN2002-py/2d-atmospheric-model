# Governing Equations

**2D Non-Hydrostatic Compressible Atmospheric Model**  
ACM40910 MSc Dissertation | University College Dublin

> All equations are implemented in `src/dynamics.py` using 2nd-order centred
> finite differences on an unstaggered grid.

---

## Table of Contents

1. [Model Description](#1-model-description)
2. [Variable Definitions](#2-variable-definitions)
3. [Physical Constants](#3-physical-constants)
4. [Thermodynamic Relations](#4-thermodynamic-relations)
5. [From 3D to 2D](#5-from-3d-to-2d)
6. [Base State and Perturbation Splitting](#6-base-state-and-perturbation-splitting)
7. [Governing Equations](#7-governing-equations)
8. [Operator Splitting](#8-operator-splitting)
9. [Spatial Discretisation](#9-spatial-discretisation)
10. [Time Integration Schemes](#10-time-integration-schemes)
11. [Initial Conditions](#11-initial-conditions)
12. [References](#12-references)

---

## 1. Model Description

The model solves the **non-hydrostatic compressible Euler equations** in
**theta-pi form** on a two-dimensional vertical slice in the $(x, z)$ plane.

Key assumptions:

- **Non-hydrostatic** — vertical accelerations are fully retained
- **Compressible** — acoustic waves are resolved; density varies
- **Dry and inviscid** — no moisture, no rotation, no viscosity
- **Two-dimensional** — all $\partial/\partial y = 0$

---

## 2. Variable Definitions

| Symbol | Description | Units |
|:---:|---|:---:|
| $u$ | Horizontal velocity | $\mathrm{m\,s^{-1}}$ |
| $w$ | Vertical velocity | $\mathrm{m\,s^{-1}}$ |
| $\theta$ | Potential temperature | K |
| $\pi$ | Exner pressure (non-dimensional) | — |
| $T$ | Absolute temperature | K |
| $p$ | Pressure | Pa |
| $\rho$ | Density | $\mathrm{kg\,m^{-3}}$ |
| $\bar{\theta}(z)$ | Base-state potential temperature | K |
| $\bar{\pi}(z)$ | Base-state Exner pressure | — |
| $\theta'$ | Potential temperature perturbation: $\theta - \bar{\theta}$ | K |
| $\pi'$ | Exner pressure perturbation: $\pi - \bar{\pi}$ | — |

---

## 3. Physical Constants

| Symbol | Value | Description | Units |
|:---:|:---:|---|:---:|
| $g$ | $9.81$ | Gravitational acceleration | $\mathrm{m\,s^{-2}}$ |
| $c_p$ | $1004$ | Specific heat at constant pressure | $\mathrm{J\,kg^{-1}\,K^{-1}}$ |
| $c_v$ | $717$ | Specific heat at constant volume | $\mathrm{J\,kg^{-1}\,K^{-1}}$ |
| $R$ | $287$ | Dry air gas constant | $\mathrm{J\,kg^{-1}\,K^{-1}}$ |
| $p_0$ | $10^5$ | Reference pressure | Pa |
| $\kappa$ | $R/c_p \approx 0.286$ | Poisson exponent | — |
| $\gamma$ | $c_p/c_v \approx 1.4$ | Ratio of specific heats | — |

---

## 4. Thermodynamic Relations

**Exner pressure** — non-dimensional form of pressure:

$$\pi = \left(\frac{p}{p_0}\right)^{R/c_p}$$

**Potential temperature** — temperature a parcel would have if brought
adiabatically to reference pressure $p_0$:

$$\theta = \frac{T}{\pi}$$

**Equation of state** for dry air in theta-pi form:

$$p = p_0 \, \pi^{c_p/R}$$

**Density** recovered from the ideal gas law:

$$\rho = \frac{p_0}{R \, \theta} \, \pi^{c_v/R}$$

---

## 5. From 3D to 2D

### 5.1 Full 3D equations (Kalnay Ch. 2–3)

The three-dimensional momentum equation in Eulerian form (from your lecture notes):

$$\frac{Du}{Dt} = \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z}$$

This is the **material derivative** — rate of change following a fluid parcel.
The $\partial/\partial t$ term is Eulerian; the remaining terms are **advection**.

### 5.2 Reduction to the x-z plane

Setting $\partial/\partial y = 0$ and dropping the $v$-equation:

$$\frac{Du}{Dt} \longrightarrow \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + w\frac{\partial u}{\partial z}$$

---

## 6. Base State and Perturbation Splitting

### 6.1 Motivation

The atmosphere is never truly at rest. There is always a background pressure
and temperature varying with height. To avoid large cancelling terms numerically,
all prognostic variables are split into a background **base state** (overbar)
plus a **perturbation** (prime):

$$\theta(x,z,t) = \bar{\theta}(z) + \theta'(x,z,t)$$

$$\pi(x,z,t) = \bar{\pi}(z) + \pi'(x,z,t)$$

### 6.2 Base-state requirements

The base state $\bar{\theta}(z)$, $\bar{\pi}(z)$ must be:
- Horizontally uniform — a function of $z$ only
- Time-invariant — does not change during the simulation
- In **hydrostatic balance**:

$$c_p \, \bar{\theta} \, \frac{d\bar{\pi}}{dz} = -g$$

### 6.3 Isothermal base state

For benchmark cases, a constant background temperature $T_0 = 300\ \mathrm{K}$
is assumed. Hydrostatic balance and the ideal gas law then give:

**Background pressure** (exponential decay with height):

$$\bar{p}(z) = p_0 \exp\left(\frac{-g \, z}{R \, T_0}\right)$$

**Background Exner pressure:**

$$\bar{\pi}(z) = \left(\frac{\bar{p}(z)}{p_0}\right)^{R/c_p}$$

**Background potential temperature:**

$$\bar{\theta}(z) = \frac{T_0}{\bar{\pi}(z)}$$

> **Zero amplitude test:** If nothing is added to this state — that is, if
> $\theta' = \pi' = u = w = 0$ everywhere — the solution should remain
> perfectly still for all time. This is the first numerical validation test.

---

## 7. Governing Equations

**Source:** Equation set 1 of Giraldo & Restelli (2008), as specified by Dr Clancy.

---

### Equation 1 — Horizontal Momentum

$$\frac{\partial u}{\partial t} = -u\frac{\partial u}{\partial x} - w\frac{\partial u}{\partial z} - c_p \left(\bar{\theta} + \theta'\right) \frac{\partial \pi'}{\partial x}$$

| Term | Physical meaning |
|---|---|
| $-u\,\dfrac{\partial u}{\partial x} - w\,\dfrac{\partial u}{\partial z}$ | Advection of $u$ by the flow |
| $-c_p(\bar{\theta}+\theta')\,\dfrac{\partial \pi'}{\partial x}$ | Horizontal pressure gradient force |

---

### Equation 2 — Vertical Momentum

$$\frac{\partial w}{\partial t} = -u\frac{\partial w}{\partial x} - w\frac{\partial w}{\partial z} - c_p \left(\bar{\theta} + \theta'\right) \frac{\partial \pi'}{\partial z} + g\frac{\theta'}{\bar{\theta}}$$

| Term | Physical meaning |
|---|---|
| $-u\,\dfrac{\partial w}{\partial x} - w\,\dfrac{\partial w}{\partial z}$ | Advection of $w$ by the flow |
| $-c_p(\bar{\theta}+\theta')\,\dfrac{\partial \pi'}{\partial z}$ | Vertical pressure gradient force |
| $g\,\dfrac{\theta'}{\bar{\theta}}$ | **Buoyancy** — warm air ($\theta' > 0$) accelerates upward |

---

### Equation 3 — Exner Pressure Perturbation

$$\frac{\partial \pi'}{\partial t} = -u\frac{\partial \pi'}{\partial x} - w\frac{\partial \pi'}{\partial z} - \frac{R}{c_v}\left(\bar{\pi} + \pi'\right)\left(\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z}\right) + \frac{g \, w}{c_p \, \bar{\theta}}$$

| Term | Physical meaning |
|---|---|
| $-u\,\dfrac{\partial \pi'}{\partial x} - w\,\dfrac{\partial \pi'}{\partial z}$ | Advection of $\pi'$ |
| $-\dfrac{R}{c_v}(\bar{\pi}+\pi')\left(\dfrac{\partial u}{\partial x}+\dfrac{\partial w}{\partial z}\right)$ | **Acoustic source** — flow divergence changes pressure |
| $\dfrac{g\,w}{c_p\,\bar{\theta}}$ | Background term from base-state gradient |

> **Stiffness note:** The acoustic source term generates fast acoustic waves
> travelling at $c_s = \sqrt{\gamma R T_0} \approx 340\ \mathrm{m\,s^{-1}}$.
> Convective motions of interest travel at only $\sim 10$–$30\ \mathrm{m\,s^{-1}}$.
> Explicit schemes must satisfy $c_s \, \Delta t / \Delta x < 1$, forcing very
> small time steps. Semi-implicit and exponential schemes remove this constraint.

---

### Equation 4 — Potential Temperature Perturbation

$$\frac{\partial \theta'}{\partial t} = -u\frac{\partial \theta'}{\partial x} - w\frac{\partial \theta'}{\partial z} - w\frac{d\bar{\theta}}{dz}$$

| Term | Physical meaning |
|---|---|
| $-u\,\dfrac{\partial \theta'}{\partial x} - w\,\dfrac{\partial \theta'}{\partial z}$ | Advection of $\theta'$ by the flow |
| $-w\,\dfrac{d\bar{\theta}}{dz}$ | Vertical motion through the background temperature gradient |

---

## 8. Operator Splitting

Writing the state vector $\mathbf{q} = (u,\, w,\, \theta',\, \pi')^\top$,
the system of equations (1)–(4) takes the compact form:

$$\frac{\partial \mathbf{q}}{\partial t} = \mathcal{L}\,\mathbf{q} + \mathcal{N}(\mathbf{q})$$

where:

- $\mathcal{L}$ is the **linear operator** containing the acoustic terms,
  buoyancy, and base-state gradient terms
- $\mathcal{N}(\mathbf{q})$ is the **nonlinear operator** containing all
  advection terms $-u\,\partial/\partial x - w\,\partial/\partial z$

This splitting is the foundation for the advanced time schemes:

**Semi-implicit** — treat $\mathcal{L}$ implicitly, $\mathcal{N}$ explicitly:

$$\frac{\mathbf{q}^{n+1} - \mathbf{q}^n}{\Delta t} = \frac{1}{2}\mathcal{L}\left(\mathbf{q}^{n+1} + \mathbf{q}^n\right) + \mathcal{N}(\mathbf{q}^n)$$

**Exponential (EPI)** — compute $e^{\mathcal{L}\Delta t}$ exactly via Krylov:

$$\mathbf{q}^{n+1} = e^{\mathcal{L}\Delta t}\,\mathbf{q}^n + \Delta t\,\varphi_1(\mathcal{L}\Delta t)\,\mathcal{N}(\mathbf{q}^n) + \cdots$$

where $\varphi_1(z) = (e^z - 1)/z$ is the first phi-function, approximated
using the phipm Krylov algorithm (Pudykiewicz & Clancy 2022).

---

## 9. Spatial Discretisation

### 9.1 Grid

An **unstaggered grid** is used — all variables $(u, w, \theta', \pi')$ share
the same set of cell-centre points:

$$x_i = \left(i + \frac{1}{2}\right)\Delta x, \qquad i = 0, 1, \ldots, n_x - 1$$

$$z_k = \left(k + \frac{1}{2}\right)\Delta z, \qquad k = 0, 1, \ldots, n_z - 1$$

All variable arrays have shape $(n_z,\, n_x)$.

> An Arakawa C-grid was tested initially but rejected. Boundary condition
> control on the unstaggered grid was found to be the dominant factor in
> solution quality (Giraldo & Restelli 2008, Fig. 1).

### 9.2 Finite-Difference Stencils

**Interior — 2nd-order centred difference:**

$$\frac{\partial f}{\partial x}\bigg|_{k,\,i} \approx \frac{f_{k,\,i+1} - f_{k,\,i-1}}{2\,\Delta x} + \mathcal{O}(\Delta x^2)$$

$$\frac{\partial f}{\partial z}\bigg|_{k,\,i} \approx \frac{f_{k+1,\,i} - f_{k-1,\,i}}{2\,\Delta z} + \mathcal{O}(\Delta z^2)$$

**Bottom boundary** ($k = 0$) — 1st-order forward difference:

$$\frac{\partial f}{\partial z}\bigg|_{0,\,i} \approx \frac{f_{1,\,i} - f_{0,\,i}}{\Delta z} + \mathcal{O}(\Delta z)$$

**Top boundary** ($k = n_z - 1$) — 1st-order backward difference:

$$\frac{\partial f}{\partial z}\bigg|_{n_z-1,\,i} \approx \frac{f_{n_z-1,\,i} - f_{n_z-2,\,i}}{\Delta z} + \mathcal{O}(\Delta z)$$

**$x$-direction** — periodic boundary conditions:

$$f_{k,\,-1} \equiv f_{k,\,n_x-1}, \qquad f_{k,\,n_x} \equiv f_{k,\,0}$$

Implemented in code via `numpy.roll`.

### 9.3 Boundary Conditions

| Boundary | Condition | Reason |
|---|---|---|
| $x = 0$ and $x = L_x$ | Periodic: $f(0, z) = f(L_x, z)$ | Repeating horizontal domain |
| $z = 0$ — bottom | $w = 0$,  $\partial\theta'/\partial t = 0$ | Solid impermeable floor |
| $z = L_z$ — top | $w = 0$,  $\partial\theta'/\partial t = 0$ | Solid impermeable lid |

### 9.4 Rayleigh Sponge Layer

A damping layer in the top 20% of the domain ($z \geq z_s = 0.8\,L_z$)
prevents upward-propagating waves from reflecting off the rigid lid.
The modified tendency equation becomes:

$$\frac{\partial f}{\partial t} = \mathrm{RHS}(f) - \alpha(z)\, f'$$

The damping coefficient ramps smoothly from zero using a $\sin^2$ profile:

$$\alpha(z) = \alpha_{\max} \sin^2\!\left(\frac{\pi}{2} \cdot \frac{z - z_s}{L_z - z_s}\right), \qquad z \geq z_s$$

with $\alpha_{\max} = 0.01\ \mathrm{s^{-1}}$ as default.

---

## 10. Time Integration Schemes

### Scheme 1 — FTCS (Forward Euler)

$$\phi^{n+1} = \phi^{n} + \Delta t\, F(\phi^{n})$$

First-order accurate. Amplification factor $A = 1 + \lambda\Delta t$.
Unstable for oscillatory modes — expect blow-up for large $\Delta t$.

---

### Scheme 2 — BTCS (Backward Euler, linearised)

True BTCS solves $\phi^{n+1} = \phi^n + \Delta t\,F(\phi^{n+1})$ implicitly.
Implemented here as one Picard (predictor-corrector) iteration:

$$\phi^{*} = \phi^n + \Delta t\, F(\phi^n)$$

$$\phi^{n+1} = \phi^n + \Delta t\, F(\phi^{*})$$

More stable than FTCS. Full implicit solve to be added with the semi-implicit scheme.

---

### Scheme 3 — CTCS (Leapfrog)

$$\phi^{n+1} = \phi^{n-1} + 2\,\Delta t\, F(\phi^{n})$$

Second-order accurate. Neutral amplitude $|A| = 1$ for oscillatory modes.
Has an undamped **computational mode** — controlled using the
**Robert–Asselin filter** applied after each step:

$$\tilde{\phi}^{n} = \phi^{n} + \alpha_f \left(\phi^{n-1} - 2\phi^{n} + \phi^{n+1}\right), \qquad \alpha_f \approx 0.1$$

> This is the scheme used by Robert (1993) for the warm bubble benchmark.
> Note: the filter reduces accuracy from 2nd to 1st order.

---

### Scheme 4 — RK4 (Classical Runge–Kutta)

$$k_1 = F(\phi^n)$$

$$k_2 = F\!\left(\phi^n + \frac{\Delta t}{2}\,k_1\right)$$

$$k_3 = F\!\left(\phi^n + \frac{\Delta t}{2}\,k_2\right)$$

$$k_4 = F\!\left(\phi^n + \Delta t\,k_3\right)$$

$$\phi^{n+1} = \phi^n + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

Fourth-order accurate. Maximum stable $\omega\Delta t \approx 2.82$.
Amplitude error $\mathcal{O}[(\omega\Delta t)^6]$. Primary explicit scheme.

---

### Planned Schemes

| Scheme | Key property | Reference |
|---|---|---|
| Semi-implicit | Removes acoustic CFL constraint | Klemp & Wilhelmson (1978) |
| EPI2 | 2nd-order exponential via Krylov | Pudykiewicz & Clancy (2022) |
| EPI3 | 3rd-order exponential with correction | Pudykiewicz & Clancy (2022) |

---

## 11. Initial Conditions

### 11.1 Zero Amplitude Test

$$u = w = \theta' = \pi' = 0 \qquad \text{everywhere at } t = 0$$

Expected: state remains identically zero for all $t > 0$.
Any growth indicates a numerical error. All four schemes pass this test.

### 11.2 Warm Bubble — Robert (1993)

A Gaussian thermal perturbation centred at $(x_c,\, z_c)$:

$$\theta'(x,\, z,\, 0) = A \exp\!\left(-\frac{(x - x_c)^2 + (z - z_c)^2}{r^2}\right)$$

All other perturbation variables are zero initially.

| Parameter | Value |
|:---:|---|
| $A$ | $2\ \mathrm{K}$ |
| $x_c$ | $L_x / 2$ |
| $z_c$ | $0.4\, L_z$ |
| $r$ | $250\ \mathrm{m}$ |
| $L_x = L_z$ | $5000\ \mathrm{m}$ |
| $\Delta x = \Delta z$ | $20\ \mathrm{m}$ |
| Integration time | $30\ \mathrm{min}$ |

### 11.3 Cold Density Current — Straka et al. (1993)

Cold elliptical perturbation. Parameters and implementation details to be added
when this benchmark is implemented.

---

## 12. References

1. Giraldo, F.X. & Restelli, M. (2008). A study of spectral element and discontinuous Galerkin methods for the Euler and Navier–Stokes equations. *Journal of Computational Physics*, **227**, 3849–3877.

2. Pudykiewicz, J.A. & Clancy, C. (2022). Exponential time integration for the compressible Euler equations. *Journal of Computational Physics*, **449**, 110803.

3. Robert, A. (1993). Bubble convection experiments with a semi-implicit formulation of the Euler equations. *Journal of the Atmospheric Sciences*, **50**(13), 1865–1873.

4. Straka, J.M., Wilhelmson, R.B., Wicker, L.J., Anderson, J.R. & Droegemeier, K.K. (1993). Numerical solutions of a non-linear density current: A benchmark solution and comparisons. *International Journal for Numerical Methods in Fluids*, **17**, 1–22.

5. Durran, D.R. (2010). *Numerical Methods for Fluid Dynamics: With Applications to Geophysics*, 2nd ed. Springer.

6. Kalnay, E. (2003). *Atmospheric Modelling, Data Assimilation and Predictability*. Cambridge University Press.

7. Klemp, J.B. & Wilhelmson, R.B. (1978). The simulation of three-dimensional convective storm dynamics. *Journal of the Atmospheric Sciences*, **35**, 1070–1096.
