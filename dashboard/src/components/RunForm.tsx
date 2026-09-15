import { useState } from "react";
import { ALL_SCHEMES, BOUNDS, type RunRequest, type SchemeName } from "../types";

interface Props {
  onSubmit: (req: Partial<RunRequest>) => void;
  submitting: boolean;
}

export function RunForm({ onSubmit, submitting }: Props) {
  const [scheme, setScheme] = useState<SchemeName>("RK4");
  const [dx, setDx] = useState<number>(BOUNDS.dx.default);
  const [tEnd, setTEnd] = useState<number>(BOUNDS.t_end.default);
  const [bubbleAmp, setBubbleAmp] = useState<number>(BOUNDS.bubble_amp.default);
  const [bubbleR, setBubbleR] = useState<number>(BOUNDS.bubble_r.default);
  const [diffusion, setDiffusion] = useState<number>(BOUNDS.diffusion.default);
  const [shapiro, setShapiro] = useState(false);
  const [useAutoDt, setUseAutoDt] = useState(true);
  const [dt, setDt] = useState<number>(0.02);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      scheme,
      dx,
      dt: useAutoDt ? null : dt,
      t_end: tEnd,
      bubble_amp: bubbleAmp,
      bubble_r: bubbleR,
      diffusion,
      shapiro,
    });
  }

  return (
    <form className="run-form" onSubmit={handleSubmit}>
      <h2>New run</h2>

      <label>
        Scheme
        <select value={scheme} onChange={(e) => setScheme(e.target.value as SchemeName)}>
          {ALL_SCHEMES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>

      <label>
        Grid spacing Δx ({BOUNDS.dx.min}–{BOUNDS.dx.max} m)
        <input
          type="number"
          min={BOUNDS.dx.min}
          max={BOUNDS.dx.max}
          step={5}
          value={dx}
          onChange={(e) => setDx(Number(e.target.value))}
        />
      </label>

      <label className="checkbox-row">
        <input type="checkbox" checked={useAutoDt} onChange={(e) => setUseAutoDt(e.target.checked)} />
        Auto Δt (recommended — matches the validated benchmark formula)
      </label>
      {!useAutoDt && (
        <label>
          Δt ({BOUNDS.dt.min}–{BOUNDS.dt.max} s)
          <input
            type="number"
            min={BOUNDS.dt.min}
            max={BOUNDS.dt.max}
            step={0.01}
            value={dt}
            onChange={(e) => setDt(Number(e.target.value))}
          />
        </label>
      )}

      <label>
        End time t_end ({BOUNDS.t_end.min}–{BOUNDS.t_end.max} s)
        <input
          type="number"
          min={BOUNDS.t_end.min}
          max={BOUNDS.t_end.max}
          step={5}
          value={tEnd}
          onChange={(e) => setTEnd(Number(e.target.value))}
        />
      </label>

      <label>
        Bubble amplitude θ_c ({BOUNDS.bubble_amp.min}–{BOUNDS.bubble_amp.max} K)
        <input
          type="number"
          min={BOUNDS.bubble_amp.min}
          max={BOUNDS.bubble_amp.max}
          step={0.1}
          value={bubbleAmp}
          onChange={(e) => setBubbleAmp(Number(e.target.value))}
        />
      </label>

      <label>
        Bubble radius r_c ({BOUNDS.bubble_r.min}–{BOUNDS.bubble_r.max} m)
        <input
          type="number"
          min={BOUNDS.bubble_r.min}
          max={BOUNDS.bubble_r.max}
          step={10}
          value={bubbleR}
          onChange={(e) => setBubbleR(Number(e.target.value))}
        />
      </label>

      <label>
        Diffusion κ ({BOUNDS.diffusion.min}–{BOUNDS.diffusion.max} m⁴/s, 0 = inviscid)
        <input
          type="number"
          min={BOUNDS.diffusion.min}
          max={BOUNDS.diffusion.max}
          step={1}
          value={diffusion}
          onChange={(e) => setDiffusion(Number(e.target.value))}
        />
      </label>

      <label className="checkbox-row">
        <input type="checkbox" checked={shapiro} onChange={(e) => setShapiro(e.target.checked)} />
        Shapiro filter
      </label>

      <button type="submit" disabled={submitting}>
        {submitting ? "Submitting…" : "Run simulation"}
      </button>
    </form>
  );
}
