import type { RunSummary } from "../types";
import { StatusBadge } from "./StatusBadge";

interface Props {
  runs: RunSummary[];
  selectedId: string | null;
  onSelect: (runId: string) => void;
}

export function RunList({ runs, selectedId, onSelect }: Props) {
  return (
    <div className="run-list">
      <h2>Recent runs</h2>
      {runs.length === 0 && <p className="muted">No runs yet — submit one to get started.</p>}
      <ul>
        {runs.map((r) => (
          <li key={r.run_id}>
            <button
              className={`run-list-item ${r.run_id === selectedId ? "selected" : ""}`}
              onClick={() => onSelect(r.run_id)}
            >
              <span className="run-scheme">{r.request.scheme}</span>
              <span className="run-meta">
                Δx={r.request.dx}m · t_end={r.request.t_end}s
              </span>
              <StatusBadge status={r.status} />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
