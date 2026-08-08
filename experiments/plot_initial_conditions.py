"""
plot_initial_conditions.py
==========================
Generates initial condition plots for both G&R Case 2 and P&C Exp 1.
Used for presentation slides.

USAGE
-----
  python experiments/plot_initial_conditions.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# G&R Case 2 — Cosine bell bubble
# ---------------------------------------------------------------------------
LX_GR, LZ_GR = 1000.0, 1000.0
DX_GR = 10.0
NX_GR = NZ_GR = int(LX_GR / DX_GR)

x_gr = (np.arange(NX_GR) + 0.5) * DX_GR
z_gr = (np.arange(NZ_GR) + 0.5) * DX_GR
X_GR, Z_GR = np.meshgrid(x_gr, z_gr)

r_gr = np.sqrt((X_GR - 500.0)**2 + (Z_GR - 350.0)**2)
theta_gr = np.where(r_gr <= 250.0,
                    0.5 * 0.5 * (1.0 + np.cos(np.pi * r_gr / 250.0)),
                    0.0)

# ---------------------------------------------------------------------------
# P&C Exp 1 — Cylindrical + Gaussian bubble
# ---------------------------------------------------------------------------
LX_PC, LZ_PC = 5000.0, 5000.0
DX_PC = 20.0
NX_PC = NZ_PC = int(LX_PC / DX_PC)

x_pc = (np.arange(NX_PC) + 0.5) * DX_PC
z_pc = (np.arange(NZ_PC) + 0.5) * DX_PC
X_PC, Z_PC = np.meshgrid(x_pc, z_pc)

AT   = 0.5    # K
a    = 400.0  # m  (20 * dx = 20 * 20)
sigma = 100.0  # m  (5 * dx)
x0, z0 = 2500.0, 700.0

r_pc = np.sqrt((X_PC - x0)**2 + (Z_PC - z0)**2)
theta_pc = np.where(r_pc <= a,
                    AT,
                    AT * np.exp(-(r_pc - a)**2 / (2 * sigma**2)))
theta_pc[theta_pc < 0.01] = 0.0   # clip near-zero tail

# ---------------------------------------------------------------------------
# Plot: side-by-side comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

cmap = "RdYlBu_r"

# --- G&R ---
ax = axes[0]
disp_gr = gaussian_filter(theta_gr, sigma=0.8)
im = ax.imshow(disp_gr, origin="lower",
               extent=[0, LX_GR/1000, 0, LZ_GR/1000],
               cmap=cmap, vmin=0.0, vmax=0.55,
               interpolation="bicubic", aspect="equal")
# Contours
x_km = x_gr / 1000;  z_km = z_gr / 1000
Xk, Zk = np.meshgrid(x_km, z_km)
ax.contour(Xk, Zk, disp_gr,
           levels=np.arange(0.05, 0.53, 0.05),
           colors="white", linewidths=0.6, alpha=0.7)

ax.set_xlabel("x [km]", fontsize=12)
ax.set_ylabel("z [km]", fontsize=12)
ax.set_title("G&R Case 2 — Cosine Bell Bubble\n"
             r"$\theta_c = 0.5$ K,  $r_c = 250$ m,  centre $(500, 350)$ m",
             fontsize=10.5)
ax.set_facecolor("#0a1628")
plt.colorbar(im, ax=ax, label=r"$\theta'$ [K]", shrink=0.85)

# Annotations
ax.annotate("", xy=(0.5, 0.58), xytext=(0.5, 0.38),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
ax.text(0.52, 0.47, "buoyancy\ndrives rise",
        color="white", fontsize=8, va="center")

# --- P&C ---
ax = axes[1]
disp_pc = gaussian_filter(theta_pc, sigma=0.8)
im2 = ax.imshow(disp_pc, origin="lower",
                extent=[0, LX_PC/1000, 0, LZ_PC/1000],
                cmap=cmap, vmin=0.0, vmax=0.55,
                interpolation="bicubic", aspect="equal")

x_km_pc = x_pc / 1000;  z_km_pc = z_pc / 1000
Xk2, Zk2 = np.meshgrid(x_km_pc, z_km_pc)
ax.contour(Xk2, Zk2, disp_pc,
           levels=np.arange(0.05, 0.53, 0.05),
           colors="white", linewidths=0.6, alpha=0.7)

ax.set_xlabel("x [km]", fontsize=12)
ax.set_ylabel("z [km]", fontsize=12)
ax.set_title("P&C Exp 1 — Cylindrical + Gaussian Bubble\n"
             r"$A_T = 0.5$ K,  $a = 400$ m,  $\sigma = 100$ m,  centre $(2500, 700)$ m",
             fontsize=10.5)
ax.set_facecolor("#0a1628")
plt.colorbar(im2, ax=ax, label=r"$\theta'$ [K]", shrink=0.85)

ax.annotate("", xy=(2.5, 1.45), xytext=(2.5, 0.8),
            xycoords="data",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
ax.text(2.6, 1.1, "buoyancy\ndrives rise",
        color="white", fontsize=8, va="center")

fig.suptitle("Initial Conditions — Both Benchmark Cases",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ic_both_benchmarks.png"),
            dpi=160, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: output/figures/ic_both_benchmarks.png")

# ---------------------------------------------------------------------------
# Also save individual plots (for slides that need just one)
# ---------------------------------------------------------------------------
for label, theta, LX, LZ, dx, title in [
    ("ic_gr",
     theta_gr, LX_GR, LZ_GR, DX_GR,
     "G&R Case 2 — Initial Condition (t = 0 s)\n"
     r"Cosine bell: $\theta_c=0.5$ K, $r_c=250$ m, centre $(500,350)$ m"),
    ("ic_pc",
     theta_pc, LX_PC, LZ_PC, DX_PC,
     "P&C Exp 1 — Initial Condition (t = 0 s)\n"
     r"Cylindrical: $A_T=0.5$ K, $a=400$ m, centre $(2500,700)$ m"),
]:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    disp = gaussian_filter(theta, sigma=0.8)
    im = ax.imshow(disp, origin="lower",
                   extent=[0, LX/1000, 0, LZ/1000],
                   cmap=cmap, vmin=0.0, vmax=0.55,
                   interpolation="bicubic", aspect="equal")
    x_km_ = (np.arange(int(LX/dx)) + 0.5) * dx / 1000
    z_km_ = (np.arange(int(LZ/dx)) + 0.5) * dx / 1000
    Xk_, Zk_ = np.meshgrid(x_km_, z_km_)
    ax.contour(Xk_, Zk_, disp,
               levels=np.arange(0.05, 0.53, 0.05),
               colors="white", linewidths=0.6, alpha=0.7)
    ax.set_xlabel("x [km]", fontsize=12)
    ax.set_ylabel("z [km]", fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.set_facecolor("#0a1628")
    plt.colorbar(im, ax=ax, label=r"$\theta'$ [K]")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{label}.png")
    plt.savefig(path, dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {path}")
