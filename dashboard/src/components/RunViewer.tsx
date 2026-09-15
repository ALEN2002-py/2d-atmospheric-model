import { useEffect, useState } from "react";
import { getRun, snapshotUrl } from "../api";
import type { RunDetail } from "../types";
import { StatusBadge } from "./StatusBadge";

const POLL_MS = 700;

export function RunViewer({ runId }: { runId: string }) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [snapshotSrc, setSnapshotSrc] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const d = await getRun(runId);
        if (cancelled) return;
        setDetail(d);
        setFetchError(null);
        if (d.status === "running" || d.status === "done") {
          setSnapshotSrc(snapshotUrl(runId));
        }
        if (d.status === "queued" || d.status === "running") {
          timer = setTimeout(poll, POLL_MS);
        }
      } catch (err) {
        if (!cancelled) setFetchError(err instanceof Error ? err.message : String(err));
      }
    }

    setDetail(null);
    setSnapshotSrc(null);
    poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId]);

  if (fetchError) return <p className="error-text">Failed to load run: {fetchError}</p>;
  if (!detail) return <p className="muted">Loading…</p>;

  const pct = detail.progress ? Math.round((100 * detail.progress.step) / detail.progress.n_steps) : 0;

  return (
    <div className="run-viewer">
      <div className="run-viewer-header">
        <h2>
          Run {detail.run_id} — {detail.request.scheme}
        </h2>
        <StatusBadge status={detail.status} />
      </div>

      <p className="muted">
        Δx={detail.request.dx}m · t_end={detail.request.t_end}s · θ_c={detail.request.bubble_amp}K ·
        r_c={detail.request.bubble_r}m
        {detail.request.diffusion > 0 && ` · κ=${detail.request.diffusion}`}
        {detail.request.shapiro && " · Shapiro"}
      </p>

      {(detail.status === "queued" || detail.status === "running") && (
        <div className="progress-bar">
          <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
          <span className="progress-bar-label">
            {detail.progress ? `${detail.progress.step} / ${detail.progress.n_steps} steps (t=${detail.progress.t.toFixed(1)}s)` : "Starting…"}
          </span>
        </div>
      )}

      {detail.status === "error" && <p className="error-text">Error: {detail.error}</p>}

      {snapshotSrc && (
        <img className="snapshot" src={snapshotSrc} alt="theta' field snapshot" />
      )}

      {detail.result && (
        <table className="result-table">
          <tbody>
            <tr>
              <td>t_final</td>
              <td>{detail.result.t_final.toFixed(2)} s</td>
            </tr>
            <tr>
              <td>θ'_max / θ'_min</td>
              <td>
                {detail.result.theta_max.toFixed(4)} / {detail.result.theta_min.toFixed(4)} K
              </td>
            </tr>
            <tr>
              <td>w_max / w_min</td>
              <td>
                {detail.result.w_max.toFixed(4)} / {detail.result.w_min.toFixed(4)} m/s
              </td>
            </tr>
            <tr>
              <td>u_max / u_min</td>
              <td>
                {detail.result.u_max.toFixed(4)} / {detail.result.u_min.toFixed(4)} m/s
              </td>
            </tr>
            <tr>
              <td>Wall time</td>
              <td>{detail.result.wall_time_s.toFixed(2)} s</td>
            </tr>
            <tr>
              <td>Blown up?</td>
              <td>{detail.result.blown_up ? "Yes" : "No"}</td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}
