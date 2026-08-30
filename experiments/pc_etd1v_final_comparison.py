"""
pc_etd1v_final_comparison.py
=============================
Combines the per-variant best-tuned ETD1V diffusion runs (each found via the
kappa-scale sweep in pc_etd1v_diffusion_comparison.py) into one consolidated
5-panel comparison for the writeup:

  IDEAL   -- kappa-scale=1 (no diffusion; blows up t=1170s, no tuning applies)
  nabla2  -- kappa-scale=4   (clean: theta_max=0.482K)
  nabla4  -- kappa-scale=16  (best achievable, plateaued: theta_max=0.891K)
  nabla8  -- kappa-scale=64  (best achievable, plateaued: theta_max=1.093K)
  Shapiro -- kappa-scale=1   (diffusion-independent; clean: theta_max=0.478K)

Reads each variant's cached results directly (no re-simulation) and reuses
pc_etd1v_diffusion_comparison.py's own plotting functions.

USAGE
-----
  python experiments/pc_etd1v_final_comparison.py
"""

import os
import pickle
import sys
import types

if getattr(sys.stdout, "encoding", None) and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def _load_src(name, path):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    exec(compile(source, os.path.abspath(path), "exec"), mod.__dict__)
    return mod

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_load_src("grid",        os.path.join(_src, "grid.py"))
_load_src("dynamics",    os.path.join(_src, "dynamics.py"))
_load_src("integrators", os.path.join(_src, "integrators.py"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pc_etd1v_diffusion_comparison as base

RESULTS_DIR = os.path.join("output", "results")
OUT_DIR     = base.OUT_DIR

# (cache file, label key inside that cache's `labels` list, display label,
#  kappa-scale used to find it -- for the printed table only)
SOURCES = [
    ("pc_etd1v_diffcomp_dx20m_cache.pkl",              "IDEAL  (no diffusion)",       "IDEAL  (no diffusion)",        1),
    ("pc_etd1v_diffcomp_dx20m_kscale4_nabla2_cache.pkl", "nabla2  (k2=3.2 m2/s)",      "nabla2  (k2=3.2 m2/s, x4)",     4),
    ("pc_etd1v_diffcomp_dx20m_kscale16_nabla4_cache.pkl","nabla4  (k4=5.3e+02 m4/s)",  "nabla4  (k4=530 m4/s, x16)",   16),
    ("pc_etd1v_diffcomp_dx20m_kscale64_nabla8_cache.pkl","nabla8  (k8=3.45e+06 m8/s)", "nabla8  (k8=3.45e6 m8/s, x64)",64),
    ("pc_etd1v_diffcomp_dx20m_cache.pkl",              "Shapiro (every 30 s)",        "Shapiro (every 30 s)",          1),
]

def main():
    all_snaps  = []
    all_stats  = []
    all_labels = []

    for cache_file, src_label, disp_label, kscale in SOURCES:
        path = os.path.join(RESULTS_DIR, cache_file)
        with open(path, "rb") as f:
            cache = pickle.load(f)
        idx = cache["labels"].index(src_label)
        all_snaps.append(cache["snaps"][idx])
        all_stats.append(cache["stats"][idx])
        all_labels.append(disp_label)
        print(f"  Loaded {disp_label:<32} from {cache_file}  "
              f"(kappa-scale={kscale})", flush=True)

    base.print_summary_table(all_labels, all_stats, 20.0)

    tag = "dx20m_final"
    base.plot_evolution_grid(all_snaps, all_labels, 20.0,
                             os.path.join(OUT_DIR, f"pc_etd1v_diffcomp_evolution_{tag}.png"))
    base.plot_final_snapshot(all_snaps, all_labels, 20.0,
                             os.path.join(OUT_DIR, f"pc_etd1v_diffcomp_final_{tag}.png"))
    base.plot_vertical_profile(all_snaps, all_labels, 20.0,
                               os.path.join(OUT_DIR, f"pc_etd1v_diffcomp_vprofile_{tag}.png"))


if __name__ == "__main__":
    main()
