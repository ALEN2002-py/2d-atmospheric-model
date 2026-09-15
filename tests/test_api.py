"""Tests for the FastAPI wrapper (api/). Uses tiny, fast configs (coarse
dx, short t_end) so the suite stays quick -- these are not accuracy tests,
that's what test_integrators.py's zero-amplitude tests and the experiments/
benchmark scripts are for."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def _wait_for_completion(run_id, timeout_s=30):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise TimeoutError(f"run {run_id} did not finish within {timeout_s}s")


def test_create_and_complete_run():
    # dx=100 -> 10x10 grid, t_end=1s -> a handful of RK4 steps: near-instant.
    r = client.post("/runs", json={
        "scheme": "RK4", "dx": 100.0, "t_end": 1.0,
        "bubble_amp": 0.5, "bubble_r": 250.0,
    })
    assert r.status_code == 201
    body = r.json()
    assert body["status"] in ("queued", "running", "done")
    run_id = body["run_id"]

    final = _wait_for_completion(run_id)
    assert final["status"] == "done", final
    assert final["result"]["blown_up"] is False
    assert final["result"]["theta_max"] > 0.0


def test_list_runs_includes_created_run():
    r = client.post("/runs", json={"scheme": "FTCS", "dx": 100.0, "t_end": 0.5})
    run_id = r.json()["run_id"]
    _wait_for_completion(run_id)

    r = client.get("/runs")
    assert r.status_code == 200
    ids = [item["run_id"] for item in r.json()]
    assert run_id in ids


def test_snapshot_png():
    r = client.post("/runs", json={"scheme": "RK4", "dx": 100.0, "t_end": 0.5})
    run_id = r.json()["run_id"]
    _wait_for_completion(run_id)

    r = client.get(f"/runs/{run_id}/snapshot")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_unknown_run_id_404():
    r = client.get("/runs/does-not-exist")
    assert r.status_code == 404


def test_invalid_scheme_rejected():
    r = client.post("/runs", json={"scheme": "NOT_A_SCHEME", "dx": 20.0})
    assert r.status_code == 422


def test_pathological_step_count_rejected():
    # Legal per-field (dt within [0, 20], t_end within (0, 120]) but the
    # combination implies way more than MAX_STEPS -- should fail at
    # execution time with a clear error, not hang the worker.
    r = client.post("/runs", json={
        "scheme": "RK4", "dx": 100.0, "dt": 0.001, "t_end": 100.0,
    })
    run_id = r.json()["run_id"]
    final = _wait_for_completion(run_id)
    assert final["status"] == "error"
    assert "exceeds" in final["error"]
