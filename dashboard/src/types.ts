// Mirrors api/schemas.py -- keep these two in sync by hand (no shared
// codegen step in this project; the FastAPI backend is the source of
// truth, this is just its shape on the frontend).

export const ALL_SCHEMES = [
  "FTCS", "BTCS", "CTCS", "RK4",
  "SI", "SI2", "SI2LU",
  "ETD1", "EPI3", "ETD1V",
] as const;

export type SchemeName = (typeof ALL_SCHEMES)[number];

export type RunStatus = "queued" | "running" | "done" | "error";

export interface RunRequest {
  scheme: SchemeName;
  dx: number;
  dt: number | null;
  t_end: number;
  bubble_amp: number;
  bubble_r: number;
  diffusion: number;
  shapiro: boolean;
}

export interface RunProgress {
  step: number;
  n_steps: number;
  t: number;
  t_end: number;
}

export interface RunResult {
  t_final: number;
  theta_max: number;
  theta_min: number;
  w_max: number;
  w_min: number;
  u_max: number;
  u_min: number;
  wall_time_s: number;
  blown_up: boolean;
}

export interface RunSummary {
  run_id: string;
  status: RunStatus;
  request: RunRequest;
  created_at: number;
}

export interface RunDetail extends RunSummary {
  progress: RunProgress | null;
  result: RunResult | null;
  error: string | null;
}

// Field bounds mirrored from api/schemas.py's Field(...) constraints, so
// the form can validate/clamp client-side before ever hitting the API.
export const BOUNDS = {
  dx: { min: 5, max: 100, default: 20 },
  dt: { min: 0, max: 20 },
  t_end: { min: 0, max: 120, default: 60 },
  bubble_amp: { min: 0, max: 5, default: 0.5 },
  bubble_r: { min: 50, max: 400, default: 250 },
  diffusion: { min: 0, max: 500, default: 0 },
} as const;
