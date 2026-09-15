import type { RunDetail, RunRequest, RunSummary } from "./types";

// Configurable API base URL, defaulting to the local dev server (see
// README's REST API section, §14, for how to run it).
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail =
      typeof body?.detail === "string"
        ? body.detail
        : Array.isArray(body?.detail)
          ? body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ")
          : `HTTP ${res.status}`;
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export function createRun(req: Partial<RunRequest>): Promise<RunSummary> {
  return request<RunSummary>("/runs", { method: "POST", body: JSON.stringify(req) });
}

export function getRun(runId: string): Promise<RunDetail> {
  return request<RunDetail>(`/runs/${runId}`);
}

export function listRuns(limit = 50): Promise<RunSummary[]> {
  return request<RunSummary[]>(`/runs?limit=${limit}`);
}

export function snapshotUrl(runId: string): string {
  // Cache-busted so the <img> tag actually re-fetches on every poll
  // instead of silently reusing the browser's cached first frame.
  return `${API_BASE}/runs/${runId}/snapshot?t=${Date.now()}`;
}

export function healthCheck(): Promise<{ status: string }> {
  return request<{ status: string }>("/health");
}
