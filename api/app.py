"""
api/app.py
==========
FastAPI wrapper around the 2D atmospheric model solver.

Run locally:
    uvicorn api.app:app --reload --port 8000

Endpoints
---------
    GET  /health                  liveness check
    POST /runs                    submit a bubble-rise simulation, returns immediately
    GET  /runs                    list recent runs (most recent first)
    GET  /runs/{run_id}           status / progress / result for one run
    GET  /runs/{run_id}/snapshot  latest theta' field as a PNG image

See api/schemas.py for the request/response contract and its documented
guardrails (grid size, t_end, step-count caps), and api/runner.py for the
time-integration loop, which mirrors experiments/gr_case2_benchmark.py's
already-validated logic rather than reimplementing it.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from api.runner import render_theta_png
from api.schemas import RunDetail, RunRequest, RunSummary
from api.store import get_run, list_runs, submit_run

app = FastAPI(
    title="2D Atmospheric Model API",
    description="Submit rising-thermal-bubble simulations (G&R 2008 Case 2 setup) "
                "against any of the 10 time-integration schemes implemented in src/.",
    version="1.0.0",
)

# Permissive CORS: this is a read-mostly demo API with no auth and no
# sensitive data, intended to be called from a browser-based dashboard
# (Phase 4) on a different origin/port during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _to_summary(record) -> RunSummary:
    return RunSummary(
        run_id=record.run_id, status=record.status,
        request=record.request, created_at=record.created_at,
    )


def _to_detail(record) -> RunDetail:
    return RunDetail(
        **_to_summary(record).model_dump(),
        progress=record.progress, result=record.result, error=record.error,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs", response_model=RunSummary, status_code=201)
def create_run(request: RunRequest):
    record = submit_run(request)
    return _to_summary(record)


@app.get("/runs", response_model=list[RunSummary])
def get_runs(limit: int = 50):
    return [_to_summary(r) for r in list_runs(limit=min(limit, 100))]


@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run_detail(run_id: str):
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run with id '{run_id}'")
    return _to_detail(record)


@app.get("/runs/{run_id}/snapshot")
def get_run_snapshot(run_id: str):
    record = get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run with id '{run_id}'")
    if record.latest_theta is None or record.grid is None:
        raise HTTPException(status_code=404, detail="No snapshot available yet")
    png_bytes = render_theta_png(record.latest_theta, record.grid)
    return Response(content=png_bytes, media_type="image/png")
