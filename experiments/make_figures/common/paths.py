"""Where the analysis reads from and writes to.

Three roots, each overridable by environment variable so a reader can point the
figures at their own re-run without editing any module::

    trained_models/  one directory per training run: its per-run measurements (config,
                     per-epoch CSVs, final metrics, delay snapshots), plus the weights
                     once fetched
    figures/measurements/    the recorded output of the analyses that need a GPU and a
                     dataset, so those figures redraw without either
    figures/         the figures and tables themselves, committed

Discovery globs on ``*seed<N>_*`` and takes the newest match, so a re-run shadows the
recorded run rather than colliding with it.
"""

import os

MODELS_ROOT = os.environ.get("DELREC_MODELS", "trained_models")
MEASUREMENTS_ROOT = os.environ.get("DELREC_MEASUREMENTS", os.path.join("figures", "measurements"))
FIGURES_ROOT = os.environ.get("DELREC_FIGURES", "figures")

# Alias kept so `runs()` reads naturally at the call sites.
RESULTS_ROOT = MODELS_ROOT


def runs(*parts):
    """Path to a training run, e.g. ``runs("HAR", "std_init_sweep")``."""
    return os.path.join(MODELS_ROOT, *parts)


def measurements(*parts):
    """Path to a recorded analysis result, e.g. ``measurements("HAR", "std", "jitter")``."""
    return os.path.join(MEASUREMENTS_ROOT, *parts)


def figures(dataset, *parts):
    """Output path for a benchmark's generated figures."""
    return os.path.join(FIGURES_ROOT, dataset, *parts)
