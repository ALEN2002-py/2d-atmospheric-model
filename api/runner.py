"""
api/runner.py
=============
Executes one RunRequest against the real solver in src/ and updates a
RunRecord in place as it progresses.

The time-integration loop below is deliberately the same pattern already
validated in experiments/gr_case2_benchmark.py (same auto-dt formula per
scheme, same SI2 Robert-Asselin bookkeeping, same Shapiro-filter interval
logic, same blow-up check) -- not a reimplementation, so the API produces
numbers directly comparable to the documented benchmark results.
"""

import io
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from api.schemas import RunProgress, RunRequest, RunResult
from grid import Grid
from integrators import robert_asselin_filter_time, shapiro_filter, step

# Fixed domain: the G&R (2008) Case 2 setup. Only resolution/timestep/scheme
# and the bubble's own amplitude/radius are exposed as request parameters.
DOMAIN = {"Lx": 1000.0, "Lz": 1000.0, "x_c": 500.0, "z_c": 350.0}

# Second, hard cap on total step count regardless of the dt/t_end combination
# chosen -- protects the shared worker pool even against a technically-legal
# but pathological request (e.g. a very small dt near its own upper bound
# combined with a t_end near its own upper bound).
MAX_STEPS = 20_000


class RunRejected(ValueError):
    """Raised for a legal-per-field but pathological request (e.g. too many steps)."""


@dataclass
class RunRecord:
    run_id: str
    request: RunRequest
    created_at: float
    status: str = "queued"
    progress: RunProgress | None = None
    result: RunResult | None = None
    error: str | None = None
    latest_theta: np.ndarray | None = field(default=None, repr=False)
    grid: Grid | None = field(default=None, repr=False)


def _auto_dt(scheme: str, dx: float) -> float:
    """Same auto-dt formula as experiments/gr_case2_benchmark.py's run()."""
    is_explicit = scheme in ("RK4", "FTCS", "BTCS", "CTCS")
    if is_explicit:
        return 0.01 * (dx / 10.0)   # acoustic CFL ~ 0.35, scaled with dx
    return 1.0                      # SI/SI2/SI2LU/ETD1/EPI3/ETD1V


def _make_initial_state(grid: Grid, req: RunRequest) -> dict:
    state = grid.allocate_state()
    r = np.sqrt((grid.x_2d - DOMAIN["x_c"]) ** 2 + (grid.z_2d - DOMAIN["z_c"]) ** 2)
    state["theta"] = np.where(
        r <= req.bubble_r,
        0.5 * req.bubble_amp * (1.0 + np.cos(np.pi * r / req.bubble_r)),
        0.0,
    )
    return state


def render_theta_png(theta: np.ndarray, grid: Grid) -> bytes:
    """Render a theta' contour plot to PNG bytes (in memory, no disk I/O)."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
    vmax = max(0.1, float(np.max(np.abs(theta))))
    levels = np.linspace(0.0, vmax, 21)
    cf = ax.contourf(grid.x_2d / 1000.0, grid.z_2d / 1000.0, theta,
                      levels=levels, cmap="turbo", extend="both")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title("theta' (K)", fontsize=10)
    fig.colorbar(cf, ax=ax, shrink=0.85)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def run_job(record: RunRecord) -> None:
    """
    Execute record.request and mutate record in place. Designed to be
    submitted to a ThreadPoolExecutor -- never raises; failures are
    captured into record.error / record.status = "error" so the worker
    thread never dies silently.
    """
    req = record.request
    try:
        dt = req.dt if req.dt is not None else _auto_dt(req.scheme, req.dx)
        n_steps = round(req.t_end / dt)
        if n_steps > MAX_STEPS:
            raise RunRejected(
                f"This request needs {n_steps} steps (t_end/dt), which exceeds "
                f"the server-side cap of {MAX_STEPS}. Increase dt or lower t_end."
            )
        if n_steps < 1:
            raise RunRejected("t_end/dt rounds to zero steps -- t_end is too small for this dt.")

        grid = Grid({"Lx": DOMAIN["Lx"], "Lz": DOMAIN["Lz"],
                     "dx": req.dx, "dz": req.dx,
                     "diffusion_coeff": req.diffusion})
        state = _make_initial_state(grid, req)
        record.grid = grid
        record.latest_theta = state["theta"].copy()

        is_si = req.scheme in ("SI", "SI2", "SI2LU")
        is_si2 = req.scheme in ("SI2", "SI2LU")
        is_epi = req.scheme in ("ETD1", "EPI3", "ETD1V")
        shapiro_interval = max(2, round(30.0 / dt)) if (is_si or is_epi) else 2

        record.status = "running"
        record.progress = RunProgress(step=0, n_steps=n_steps, t=0.0, t_end=req.t_end)

        t = 0.0
        t0_wall = time.perf_counter()
        state_old = None
        epi_n_prev = None
        blown_up = False

        for n in range(n_steps):
            q_nm1 = state_old if is_si2 else None

            state_new, state_old, epi_extra = step(
                state, grid, dt, scheme=req.scheme, state_old=state_old, epi_n_prev=epi_n_prev
            )

            if is_si2 and q_nm1 is not None:
                state_old = robert_asselin_filter_time(q_nm1, state_old, state_new)

            state = state_new
            if epi_extra is not None:
                epi_n_prev = epi_extra.get("n_rhs")

            if req.shapiro and (is_si or is_epi) and (n + 1) % shapiro_interval == 0:
                state = shapiro_filter(state, grid)

            t += dt

            if not np.all(np.isfinite(state["w"])):
                blown_up = True
                break

            # Update progress + latest snapshot every ~2% of the run (and
            # at least every 50 steps) so a client polling mid-run sees
            # motion without every step paying a numpy-copy cost.
            report_every = max(1, min(50, n_steps // 50 or 1))
            if (n + 1) % report_every == 0 or n == n_steps - 1:
                record.progress = RunProgress(step=n + 1, n_steps=n_steps, t=t, t_end=req.t_end)
                record.latest_theta = state["theta"].copy()

        elapsed = time.perf_counter() - t0_wall
        record.latest_theta = state["theta"].copy()
        record.result = RunResult(
            t_final=t,
            theta_max=float(np.max(state["theta"])),
            theta_min=float(np.min(state["theta"])),
            w_max=float(np.max(state["w"])),
            w_min=float(np.min(state["w"])),
            u_max=float(np.max(state["u"])),
            u_min=float(np.min(state["u"])),
            wall_time_s=elapsed,
            blown_up=blown_up,
        )
        record.status = "done"

    except Exception as exc:
        # Broad by design: this runs in a background worker thread with no
        # caller to propagate to, so every failure must be captured here
        # rather than killing the thread silently.
        record.status = "error"
        record.error = str(exc)
