<div align="center">

# 2D Non-Hydrostatic Compressible Atmospheric Model

**MSc Dissertation — Data & Computational Science**

University College Dublin &nbsp;|&nbsp; ACM40910 &nbsp;|&nbsp; Supervisor: Dr Colm Clancy (Met Éireann)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Governing Equations](#2-governing-equations)
3. [Base State](#3-base-state)
4. [Numerical Discretisation](#4-numerical-discretisation)
5. [Time Integration Schemes](#5-time-integration-schemes)
6. [Benchmark Test Cases](#6-benchmark-test-cases)
7. [Repository Structure](#7-repository-structure)
8. [Getting Started](#8-getting-started)
9. [Development Status](#9-development-status)
10. [References](#10-references)

---

## 1. Project Overview

This repository implements a **2D non-hydrostatic compressible atmospheric model**
in Python. The scientific goal is a systematic comparison of the efficiency and
accuracy of multiple time integration schemes — from simple explicit methods to
advanced exponential integrators — when applied to the compressible Euler equations.

The model is validated against two standard benchmarks from the atmospheric
modelling literature: the **warm bubble** of Robert (1993) and the
**cold density current** of Straka et al. (1993).

For the full mathematical derivation, variable definitions, and finite-difference
stencils, see [docs/equations.md](docs/equations.md).

---

## 2. Governing Equations

The model solves the **compressible Euler equations in theta-pi form** on a
2D vertical $(x, z)$ slice (Giraldo & Restelli 2008, Eq. set 1).

All prognostic variables are split into a horizontally uniform, time-invariant
base state (overbar) and a perturbation (prime):

$$\theta(x,z,t) = \bar{\theta}(z) + \theta'(x,z,t), \qquad \pi(x,z,t) = \bar{\pi}(z) + \pi'(x,z,t)$$

The four prognostic equations are:

$$\boxed{\frac{\partial u}{\partial t} = -u\frac{\partial u}{\partial x} - w\frac{\partial u}{\partial z} - c_p\!\left(\bar{\theta} + \theta'\right)\frac{\partial \pi'}{\partial x}} \tag{1}$$

$$\boxed{\frac{\partial w}{\partial t} = -u\frac{\partial w}{\partial x} - w\frac{\partial w}{\partial z} - c_p\!\left(\bar{\theta} + \theta'\right)\frac{\partial \pi'}{\partial z} + g\frac{\theta'}{\bar{\theta}}} \tag{2}$$

$$\boxed{\frac{\partial \pi'}{\partial t} = -u\frac{\partial \pi'}{\partial x} - w\frac{\partial \pi'}{\partial z} - \frac{R}{c_v}\!\left(\bar{\pi} + \pi'\right)\!\left(\frac{\partial u}{\partial x} + \frac{\partial w}{\partial z}\right) + \frac{gw}{c_p\bar{\theta}}} \tag{3}$$

$$\boxed{\frac{\partial \theta'}{\partial t} = -u\frac{\partial \theta'}{\partial x} - w\frac{\partial \theta'}{\partial z} - w\frac{d\bar{\theta}}{dz}} \tag{4}$$

where $\mathbf{u} = (u, w)^\top$ is the velocity vector, $\theta$ is the
potential temperature, and $\pi = (p/p_0)^{R/c_p}$ is the Exner pressure.

The term $g\,\theta'/\bar{\theta}$ in equation (2) is the **buoyancy force**.
The divergence term $(\partial u/\partial x + \partial w/\partial z)$ in
equation (3) is the **acoustic source** — it generates fast acoustic waves
($\sim 340\ \mathrm{m\,s^{-1}}$) that make the system stiff.

---

## 3. Base State

The base state satisfies the **hydrostatic balance equation**:

$$c_p\,\bar{\theta}\,\frac{d\bar{\pi}}{dz} = -g$$

For the benchmark cases an **isothermal** base state is used ($T_0 = 300\ \mathrm{K}$):

$$\bar{p}(z) = p_0\exp\!\left(\frac{-gz}{RT_0}\right), \qquad \bar{\pi}(z) = \left(\frac{\bar{p}(z)}{p_0}\right)^{R/c_p}, \qquad \bar{\theta}(z) = \frac{T_0}{\bar{\pi}(z)}$$

---

## 4. Numerical Discretisation

| Component | Choice |
|---|---|
| Spatial grid | **Unstaggered** — all variables on the same $(n_z \times n_x)$ grid |
| Interior $x$-derivatives | 2nd-order centred: $\displaystyle\frac{\partial f}{\partial x} \approx \frac{f_{i+1} - f_{i-1}}{2\,\Delta x}$ |
| Interior $z$-derivatives | 2nd-order centred: $\displaystyle\frac{\partial f}{\partial z} \approx \frac{f_{k+1} - f_{k-1}}{2\,\Delta z}$ |
| Boundary $z$-derivatives | 1st-order one-sided (forward at bottom, backward at top) |
| Lateral BCs | Periodic in $x$ |
| Top / bottom BCs | $w = 0$, $\;\partial\theta'/\partial t = 0$ |
| Sponge layer | Rayleigh damping $\alpha(z)$ in top 20% of domain |

> **Note on grid choice:** An Arakawa C-grid (staggered) was considered but
> rejected in favour of the unstaggered grid, which gives simpler and more
> reliable boundary condition control (Giraldo & Restelli 2008, Fig. 1).

---

## 5. Time Integration Schemes

Run `python menu.py` to select a scheme interactively. New schemes are added
to the registry in `menu.py` without changing any other code.

| # | Scheme | Update formula | Order | Status |
|:---:|---|---|:---:|:---:|
| 1 | **FTCS** — Forward Time Centred Space | $\phi^{n+1} = \phi^{n} + \Delta t\, F(\phi^{n})$ | 1st | ✅ Ready |
| 2 | **BTCS** — Backward Time Centred Space | $\phi^{n+1} = \phi^{n} + \Delta t\, F(\phi^{*})$ | 1st | ✅ Ready |
| 3 | **CTCS** — Leapfrog | $\phi^{n+1} = \phi^{n-1} + 2\Delta t\, F(\phi^{n})$ | 2nd | ✅ Ready |
| 4 | **RK4** — Classical Runge–Kutta | $\phi^{n+1} = \phi^{n} + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$ | 4th | ✅ Ready |
| 5 | **Semi-implicit** | $\mathcal{L}$ implicit, $\mathcal{N}$ explicit | 2nd | 🔜 Soon |
| 6 | **EPI2** — Exponential Propagation Iterative | $e^{\mathcal{L}\Delta t}$ via Krylov (phipm) | 2nd | 🔜 Soon |
| 7 | **EPI3** — Exponential Propagation Iterative | Higher-order correction | 3rd | 🔜 Soon |

---

## 6. Benchmark Test Cases

### 6.1 Zero amplitude test (numerical validation)

Set bubble amplitude $A = 0$, giving:

$$u = w = \theta' = \pi' = 0 \quad \text{everywhere at } t = 0$$

The solution should remain identically zero for all $t > 0$. Any growth
indicates a numerical error. **All four implemented schemes pass this test.**

### 6.2 Warm bubble — Robert (1993)

A Gaussian warm bubble is prescribed as the initial condition:

$$\theta'(x,z,0) = A\exp\!\left(-\frac{(x - x_c)^2 + (z - z_c)^2}{r^2}\right)$$

| Parameter | Value |
|---|---|
| Amplitude $A$ | $2\ \mathrm{K}$ |
| Domain $L_x \times L_z$ | $5\ \mathrm{km} \times 5\ \mathrm{km}$ |
| Resolution $\Delta x = \Delta z$ | $20\ \mathrm{m}$ |
| Bubble centre $(x_c, z_c)$ | $(L_x/2,\ 0.4\,L_z)$ |
| Bubble radius $r$ | $250\ \mathrm{m}$ |
| Integration time | $30\ \mathrm{min}$ |

### 6.3 Cold density current — Straka et al. (1993)

| Parameter | Value |
|---|---|
| Domain $L_x \times L_z$ | $51.2\ \mathrm{km} \times 6.4\ \mathrm{km}$ |
| Resolution $\Delta x = \Delta z$ | $50\ \mathrm{m}$ |
| Explicit diffusion $K$ | $75\ \mathrm{m^2\,s^{-1}}$ |
| Integration time | $15\ \mathrm{min}$ |

---

## 7. Repository Structure

```
2d-atmospheric-model/
│
├── src/
│   ├── grid.py               # Unstaggered grid, base state, sponge layer
│   ├── dynamics.py           # RHS of equations (1)–(4)
│   └── integrators.py        # FTCS, BTCS, CTCS, RK4  (+future schemes)
│
├── tests/
│   ├── test_grid.py
│   └── test_integrators.py
│
├── experiments/
│   ├── warm_bubble.py        # Robert (1993) benchmark
│   ├── density_current.py    # Straka et al. (1993) benchmark
│   └── efficiency_study.py   # Scheme comparison
│
├── notebooks/                # Jupyter notebooks for analysis and figures
│
├── docs/
│   ├── equations.md          # Full derivation with LaTeX equations
│   └── references.md         # Complete bibliography
│
├── output/                   # Model output — not tracked by git
│
├── menu.py                   # Interactive scheme selection menu
├── run_model.py              # Command-line run script
├── requirements.txt
└── README.md
```

---

## 8. Getting Started

### Installation

```bash
git clone https://github.com/ALEN2002-py/2d-atmospheric-model.git
cd 2d-atmospheric-model
pip install -r requirements.txt
```

### Interactive menu (recommended)

```bash
python menu.py
```

### Command-line examples

```bash
# Zero amplitude test — state must remain at zero
python run_model.py --scheme RK4 --bubble_amp 0.0 --dt 0.5 --n_steps 100

# Warm bubble with RK4
python run_model.py --scheme RK4 --bubble_amp 2.0 --dt 0.5 --n_steps 200

# Leapfrog (as used in Robert 1993)
python run_model.py --scheme CTCS --bubble_amp 2.0 --dt 0.5 --n_steps 200

# Demonstrate FTCS blow-up (large dt → unstable)
python run_model.py --scheme FTCS --bubble_amp 0.0 --dt 2.0 --n_steps 50
```

### Run tests

```bash
pytest tests/
```

---

## 9. Development Status

| Milestone | Status |
|---|:---:|
| Unstaggered grid, base state, sponge layer | ✅ Done |
| FTCS, BTCS, CTCS, RK4 implemented | ✅ Done |
| Zero amplitude test — all schemes pass | ✅ Done |
| Interactive scheme menu | ✅ Done |
| Warm bubble validation vs Robert (1993) | 🔄 In progress |
| Semi-implicit scheme | 🔜 Upcoming |
| EPI2 / EPI3 schemes | 🔜 Upcoming |
| Efficiency and accuracy comparison | 🔜 Upcoming |
| Cold density current benchmark | 🔜 Upcoming |
| Dissertation write-up | 🔜 Upcoming |

---

## 10. References

1. Giraldo, F.X. & Restelli, M. (2008). A study of spectral element and discontinuous Galerkin methods for the Euler and Navier–Stokes equations. *J. Comput. Phys.*, **227**, 3849–3877.
2. Pudykiewicz, J.A. & Clancy, C. (2022). Exponential time integration for the compressible Euler equations. *J. Comput. Phys.*, **449**, 110803.
3. Robert, A. (1993). Bubble convection experiments with a semi-implicit formulation of the Euler equations. *J. Atmos. Sci.*, **50**(13), 1865–1873.
4. Straka, J.M., Wilhelmson, R.B., Wicker, L.J., Anderson, J.R. & Droegemeier, K.K. (1993). Numerical solutions of a non-linear density current: A benchmark solution and comparisons. *Int. J. Numer. Methods Fluids*, **17**, 1–22.
5. Durran, D.R. (2010). *Numerical Methods for Fluid Dynamics: With Applications to Geophysics*, 2nd ed. Springer.
6. Kalnay, E. (2003). *Atmospheric Modelling, Data Assimilation and Predictability*. Cambridge University Press.

---

<div align="center">

MSc Data & Computational Science &nbsp;·&nbsp; University College Dublin &nbsp;·&nbsp; Submission: 31 August 2026

</div>
