"""
api/schemas.py
===============
Pydantic request/response models for the REST API.

Kept separate from runner.py (execution) and app.py (routing) so the
public contract is easy to read in one place.
"""

from typing import Literal

from pydantic import BaseModel, Field

ALL_SCHEMES = [
    "FTCS", "BTCS", "CTCS", "RK4",
    "SI", "SI2", "SI2LU",
    "ETD1", "EPI3", "ETD1V",
]

SchemeName = Literal[
    "FTCS", "BTCS", "CTCS", "RK4",
    "SI", "SI2", "SI2LU",
    "ETD1", "EPI3", "ETD1V",
]

RunStatus = Literal["queued", "running", "done", "error"]


class RunRequest(BaseModel):
    """
    A rising-thermal-bubble run request. The initial condition is always
    the G&R (2008) Case 2 cosine-bell bubble (see experiments/gr_case2_
    benchmark.py) on a fixed 1000m x 1000m domain -- only the numerics
    (scheme, resolution, timestep) and the bubble's own amplitude/radius
    are configurable. Every bound below is a deliberate guardrail so a
    request can't tie up the (shared, in-process) worker pool for an
    unbounded amount of time -- the full, unbounded benchmark suite is
    still available by running the scripts in experiments/ directly.
    """

    scheme: SchemeName = "RK4"
    dx: float = Field(20.0, ge=5.0, le=100.0,
                       description="Grid spacing in metres (dx=dz, unstaggered grid). "
                                   "5-100m; the domain is fixed at 1000x1000m, so this "
                                   "bounds the grid to at most 200x200 cells.")
    dt: float | None = Field(None, gt=0.0, le=20.0,
                                 description="Timestep in seconds. Omit to use the same "
                                             "auto-selected value experiments/gr_case2_"
                                             "benchmark.py uses for this scheme.")
    t_end: float = Field(60.0, gt=0.0, le=120.0,
                          description="Simulated end time in seconds (capped at 120s "
                                      "for this demo API; the paper benchmark runs to "
                                      "700s via experiments/gr_case2_benchmark.py).")
    bubble_amp: float = Field(0.5, gt=0.0, le=5.0,
                               description="Bubble amplitude theta_c [K]. G&R (2008) uses 0.5K.")
    bubble_r: float = Field(250.0, ge=50.0, le=400.0,
                             description="Bubble radius r_c [m]. G&R (2008) uses 250m.")
    diffusion: float = Field(0.0, ge=0.0, le=500.0,
                              description="Explicit nabla^4 hyperdiffusion coefficient [m^4/s]. "
                                          "0 = inviscid (as in the G&R benchmark).")
    shapiro: bool = Field(False, description="Apply the Shapiro (1-2-1) filter periodically "
                                              "(SI/SI2/SI2LU/ETD1/EPI3/ETD1V only).")


class RunProgress(BaseModel):
    step: int
    n_steps: int
    t: float
    t_end: float


class RunResult(BaseModel):
    t_final: float
    theta_max: float
    theta_min: float
    w_max: float
    w_min: float
    u_max: float
    u_min: float
    wall_time_s: float
    blown_up: bool


class RunSummary(BaseModel):
    run_id: str
    status: RunStatus
    request: RunRequest
    created_at: float


class RunDetail(RunSummary):
    progress: RunProgress | None = None
    result: RunResult | None = None
    error: str | None = None
