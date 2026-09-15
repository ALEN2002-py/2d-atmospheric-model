"""
api/store.py
============
In-memory run registry + a small worker pool.

Deliberately simple for a portfolio demo: a dict guarded by a lock, and a
bounded ThreadPoolExecutor. Documented limitations (not fixed here, since
fixing them means a different architecture, not a bug):
  - State is lost on process restart -- no persistence layer.
  - Single-process only -- doesn't scale past one API instance. A real
    production deployment would move this to a task queue (Celery/RQ) and
    a shared store (Redis/Postgres) so multiple API replicas share one
    queue instead of each holding an independent in-memory dict.
  - MAX_HISTORY bounds memory by evicting the oldest completed runs, not
    by any smarter policy (e.g. keeping a few of each scheme).
"""

import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from api.runner import RunRecord, run_job
from api.schemas import RunRequest

MAX_WORKERS = 2
MAX_HISTORY = 100

_lock = threading.Lock()
_runs: "OrderedDict[str, RunRecord]" = OrderedDict()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="sim-worker")


def submit_run(request: RunRequest) -> RunRecord:
    import time

    run_id = uuid.uuid4().hex[:12]
    record = RunRecord(run_id=run_id, request=request, created_at=time.time())

    with _lock:
        _runs[run_id] = record
        while len(_runs) > MAX_HISTORY:
            _runs.popitem(last=False)  # evict oldest

    _executor.submit(run_job, record)
    return record


def get_run(run_id: str) -> RunRecord | None:
    with _lock:
        return _runs.get(run_id)


def list_runs(limit: int = 50) -> list[RunRecord]:
    with _lock:
        records = list(_runs.values())
    return records[-limit:][::-1]  # most recent first
