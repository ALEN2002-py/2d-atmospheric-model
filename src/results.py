"""
results.py
==========
Save and load model experiments.

Python equivalent of MATLAB's .mat file is numpy's .npz format.
It stores arrays efficiently in a compressed binary file.

Additionally we store metadata (scheme, dt, n_steps, etc.) as a
JSON sidecar file so experiments are self-documenting.

Usage
-----
    from results import save_experiment, load_experiment, list_experiments

    # Save
    save_experiment("ctcs_dt002_bubble2", state, grid, metadata={
        "scheme": "CTCS", "dt": 0.02, "n_steps": 500,
        "bubble_amp": 2.0, "notes": "stable run"
    })

    # Load
    state, meta = load_experiment("ctcs_dt002_bubble2")
    print(meta)   # {'scheme': 'CTCS', 'dt': 0.02, ...}

    # List all saved experiments
    list_experiments()
"""

import numpy as np
import json
import os
from datetime import datetime

# All results saved here
RESULTS_DIR = "output/results"


def save_experiment(name, state, grid, metadata=None, snapshots=None):
    """
    Save a model experiment to disk.

    Parameters
    ----------
    name      : str   — unique experiment name, e.g. "ctcs_dt002_bubble2"
    state     : dict  — final state {'u', 'w', 'theta', 'pi'}
    grid      : Grid  — grid object (saves key parameters)
    metadata  : dict  — any extra info: scheme, dt, n_steps, notes, etc.
    snapshots : list  — optional list of (time, state) tuples for time series
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Build arrays to save ---
    save_dict = {
        "u":     state["u"],
        "w":     state["w"],
        "theta": state["theta"],
        "pi":    state["pi"],
    }

    # Save snapshots if provided (for time series plots)
    if snapshots is not None:
        times = np.array([t for t, _ in snapshots])
        save_dict["snapshot_times"] = times
        for i, (t, s) in enumerate(snapshots):
            for var in ["u", "w", "theta", "pi"]:
                save_dict[f"snap_{var}_{i:04d}"] = s[var]

    # Save arrays as compressed .npz
    npz_path = os.path.join(RESULTS_DIR, f"{name}.npz")
    np.savez_compressed(npz_path, **save_dict)

    # --- Build metadata ---
    meta = {
        "name":      name,
        "saved_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grid": {
            "Lx": grid.Lx, "Lz": grid.Lz,
            "dx": grid.dx, "dz": grid.dz,
            "nx": grid.nx, "nz": grid.nz,
        },
        "n_snapshots": len(snapshots) if snapshots else 0,
    }
    if metadata:
        meta.update(metadata)

    # Save metadata as human-readable JSON
    json_path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved: {npz_path}")
    print(f"  Meta:  {json_path}")
    return npz_path


def load_experiment(name):
    """
    Load a saved experiment.

    Returns
    -------
    state : dict  — final state {'u', 'w', 'theta', 'pi'}
    meta  : dict  — metadata (scheme, dt, notes, etc.)
    snapshots : list of (time, state) — if snapshots were saved, else None
    """
    npz_path  = os.path.join(RESULTS_DIR, f"{name}.npz")
    json_path = os.path.join(RESULTS_DIR, f"{name}.json")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No experiment named '{name}' in {RESULTS_DIR}")

    # Load arrays
    data = np.load(npz_path)

    state = {
        "u":     data["u"],
        "w":     data["w"],
        "theta": data["theta"],
        "pi":    data["pi"],
    }

    # Load metadata
    meta = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)

    # Reconstruct snapshots if they were saved
    snapshots = None
    if "snapshot_times" in data:
        times     = data["snapshot_times"]
        snapshots = []
        for i, t in enumerate(times):
            s = {var: data[f"snap_{var}_{i:04d}"]
                 for var in ["u", "w", "theta", "pi"]}
            snapshots.append((float(t), s))

    return state, meta, snapshots


def list_experiments():
    """Print a table of all saved experiments."""
    if not os.path.exists(RESULTS_DIR):
        print("  No results directory found. Run some experiments first.")
        return

    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    if not files:
        print("  No saved experiments found.")
        return

    print(f"\n  {'Name':<35} {'Scheme':<8} {'dt':>6} {'Steps':>7} {'Saved at'}")
    print("  " + "-" * 75)
    for f in sorted(files):
        path = os.path.join(RESULTS_DIR, f)
        with open(path) as fp:
            meta = json.load(fp)
        name    = meta.get("name", f[:-5])
        scheme  = meta.get("scheme", "?")
        dt      = meta.get("dt", "?")
        n_steps = meta.get("n_steps", "?")
        saved   = meta.get("saved_at", "?")
        print(f"  {name:<35} {scheme:<8} {str(dt):>6} {str(n_steps):>7}  {saved}")
    print()
