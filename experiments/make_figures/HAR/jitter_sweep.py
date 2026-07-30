"""Temporal-jitter robustness sweep for recurrent-delay SNNs on HAR.

HAR input is a continuous z-scored gyro signal (not spikes), so jitter resamples
the signal in time (see ``common.jitter``, ``mode="signal"``) rather than moving
discrete spikes as on SSC. Both inject a per-timestep Gaussian temporal perturbation
of magnitude σ (timesteps) and reduce to the identity at σ=0.

Runs over either HAR sweep (``--sweep {std,wd}``) via the shared engine
(``sweep_common``): per family a heatmap + accuracy-vs-jitter lines, plus the
fixed-vs-learned comparison figures (std sweep only. The wd sweep has no fixed runs).

Usage
-----
    python experiments/make_figures/HAR/jitter_sweep.py
    python experiments/make_figures/HAR/jitter_sweep.py --sweep wd
    python experiments/make_figures/HAR/jitter_sweep.py --plots-only
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import pandas as pd

from common.jitter import evaluate_jittered, JITTER_XLABEL
import runs as har
import sweep_common as sc

from delrec.utils import calc_metric_HAR

# ======================= TUNE THE SWEEP HERE ============================== #
# Jitter magnitudes (Gaussian std in timesteps. HAR windows are T=200). 0.0 = clean.
JITTER_MAGNITUDES = [0.0, 1.0, 2.0, 4.0, 8.0, 12.]
N_JITTER_SEEDS = 3                  # independent jitter realizations per magnitude
MODEL_SEEDS = har.SEEDS
CKPT = "best.pth"
# Recurrent forward kernel for all model rebuilds. Default 'triton_exact'. All
# kernels are numerically equivalent (see src/delrec/). Alternatives:
# 'v2' (portable pure-torch), 'eventdriven' (fast spike-sparse), None (layer default).
FORWARD_VERSION = "triton_exact"
# ========================================================================= #

JITTER_MODE = "signal"   # HAR input is a continuous signal
STEM = "jitter"
TITLE = "HAR: jitter"


def _evaluate(model, config, device, loader, sigma, pseed):
    return evaluate_jittered(loader, model, device, calc_metric_HAR, sigma, JITTER_MODE, pseed)


def make_plots(summary, out_dir, axis):
    sc.make_sweep_plots(summary, out_dir, STEM, JITTER_XLABEL, TITLE, axis=axis)


def run_sweep(out_dir, axis):
    rows_df, summary = sc.run_model_sweep(
        _evaluate, JITTER_MAGNITUDES, axis=axis, n_perturb_seeds=N_JITTER_SEEDS,
        clean_mag=0.0, seeds=MODEL_SEEDS, ckpt=CKPT, forward_version=FORWARD_VERSION)
    if summary is None:
        print("[error] no results.")
        return
    sc.save_tables(rows_df, summary, out_dir, STEM)
    make_plots(summary, out_dir, axis)
    sc.print_summary(summary, "HAR jitter sweep", axis)
    print(f"\nFigures (SVG) + CSVs written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/jitter")
    ap.add_argument("--plots-only", action="store_true",
                    help="Regenerate figures from an existing jitter_summary.csv.")
    args = ap.parse_args()
    axis = har.get_axis(args)
    out_dir = args.out_dir or axis.out_dir(STEM)
    if args.plots_only:
        summary = sc.load_summary(out_dir, STEM, axis)
        make_plots(summary, out_dir, axis)
        print(f"Figures (SVG) regenerated in {out_dir}/")
    else:
        run_sweep(out_dir, axis)


if __name__ == "__main__":
    main()
