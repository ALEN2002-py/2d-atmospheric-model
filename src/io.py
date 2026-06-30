"""
io.py
=====
Output helper utilities for the 2D atmospheric model.

Currently a thin wrapper around matplotlib setup that is shared by
all experiment scripts. Centralising these helpers avoids duplicating
the "Agg backend + output directory" boilerplate in every file.

Functions
---------
setup_output(subdir)
    Ensure output/figures/<subdir> exists and return the path.

ensure_agg()
    Switch matplotlib to the non-interactive Agg backend so that
    figures can be saved without a display (e.g. on HPC or headless
    containers). Call this BEFORE importing matplotlib.pyplot.

Notes
-----
The module is intentionally small. Heavier I/O (saving/loading
experiment data) is handled by results.py.
"""

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before pyplot import


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

BASE_OUTPUT = "output"
FIGURES_DIR = os.path.join(BASE_OUTPUT, "figures")
RESULTS_DIR = os.path.join(BASE_OUTPUT, "results")


def setup_output(subdir=None):
    """
    Create and return the path to an output/figures subdirectory.

    Parameters
    ----------
    subdir : str or None
        Optional subdirectory name inside output/figures/.
        If None, returns the top-level output/figures/ path.

    Returns
    -------
    path : str   — absolute-ish path to the output directory

    Example
    -------
        out = setup_output("gr_case2")
        plt.savefig(os.path.join(out, "bubble_t700.png"))
    """
    if subdir:
        path = os.path.join(FIGURES_DIR, subdir)
    else:
        path = FIGURES_DIR
    os.makedirs(path, exist_ok=True)
    return path


def ensure_agg():
    """
    Switch matplotlib to the non-interactive Agg backend.

    Safe to call multiple times (subsequent calls are no-ops if
    the backend is already set).  Must be called before the first
    import of matplotlib.pyplot.
    """
    matplotlib.use("Agg")
