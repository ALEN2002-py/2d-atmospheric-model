import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { ApiError, createRun, healthCheck, listRuns } from "./api";
import { RunForm } from "./components/RunForm";
import { RunList } from "./components/RunList";
import { RunViewer } from "./components/RunViewer";
import type { RunRequest, RunSummary } from "./types";

const LIST_POLL_MS = 2000;

function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  const refreshList = useCallback(async () => {
    try {
      const data = await listRuns();
      setRuns(data);
      setApiUp(true);
    } catch {
      setApiUp(false);
    }
  }, []);

  useEffect(() => {
    healthCheck()
      .then(() => setApiUp(true))
      .catch(() => setApiUp(false));
    refreshList();
    const timer = setInterval(refreshList, LIST_POLL_MS);
    return () => clearInterval(timer);
  }, [refreshList]);

  async function handleSubmit(req: Partial<RunRequest>) {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const summary = await createRun(req);
      setSelectedId(summary.run_id);
      await refreshList();
    } catch (err) {
      setSubmitError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>2D Atmospheric Model — Dashboard</h1>
        <p className="muted">
          Rising thermal bubble (G&amp;R 2008 Case 2), driven live against the{" "}
          <a href="https://github.com/ALEN2002-py/2d-atmospheric-model" target="_blank" rel="noreferrer">
            REST API
          </a>
          .
        </p>
        {apiUp === false && (
          <p className="error-banner">
            Can't reach the API. Start it with <code>uvicorn api.app:app --reload --port 8000</code> (see
            README §14).
          </p>
        )}
      </header>

      <main className="app-body">
        <aside className="app-sidebar">
          <RunForm onSubmit={handleSubmit} submitting={submitting} />
          {submitError && <p className="error-text">{submitError}</p>}
          <RunList runs={runs} selectedId={selectedId} onSelect={setSelectedId} />
        </aside>

        <section className="app-main">
          {selectedId ? (
            <RunViewer runId={selectedId} />
          ) : (
            <p className="muted">Select a run, or submit a new one, to see it here.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
