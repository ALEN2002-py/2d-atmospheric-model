import type { RunStatus } from "../types";

const LABELS: Record<RunStatus, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  error: "Error",
};

export function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`status-badge status-${status}`}>{LABELS[status]}</span>;
}
