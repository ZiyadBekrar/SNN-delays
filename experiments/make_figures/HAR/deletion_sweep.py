"""Sample-deletion robustness sweep for recurrent-delay SNNs on HAR.

HAR input is a continuous z-scored gyro signal (not spikes), so deletion zeroes
each (timestep, channel) value independently with probability p (see
``common.deletion``, ``mode="signal"``) rather than dropping discrete spikes as on
SSC. Each scalar reading is the deletable unit (the analog of one spike), and both
reduce to the identity at p=0.

Runs over either HAR sweep (``--sweep {std,wd}``) via the shared engine
(``sweep_common``): per family a heatmap + accuracy-vs-deletion lines, plus the
fixed-vs-learned comparison figures (std sweep only. The wd sweep has no fixed runs).

Usage
-----
    python experiments/make_figures/HAR/deletion_sweep.py
    python experiments/make_figures/HAR/deletion_sweep.py --sweep wd
    python experiments/make_figures/HAR/deletion_sweep.py --plots-only
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import pandas as pd

from common.deletion import evaluate_deleted, DELETION_XLABELS
import runs as har
import sweep_common as sc

from delrec.utils import calc_metric_HAR

# ======================= TUNE THE SWEEP HERE ============================== #
# Per-value deletion probabilities in [0, 1]. 0.0 = clean input.
DELETION_PROBS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_DELETION_SEEDS = 3                # independent deletion realizations per probability
MODEL_SEEDS = har.SEEDS
CKPT = "best.pth"
# Recurrent forward kernel for all model rebuilds. Default 'triton_exact'. All
# kernels are numerically equivalent (see src/delrec/). Alternatives:
# 'v2' (portable pure-torch), 'eventdriven' (fast spike-sparse), None (layer default).
FORWARD_VERSION = "triton_exact"
# ========================================================================= #

DELETION_MODE = "signal"   # HAR input is a continuous signal
STEM = "deletion"
TITLE = "HAR: deletion"


def _evaluate(model, config, device, loader, p, pseed):
    return evaluate_deleted(loader, model, device, calc_metric_HAR, p, DELETION_MODE, pseed)


def make_plots(summary, out_dir, axis):
    sc.make_sweep_plots(summary, out_dir, STEM, DELETION_XLABELS[DELETION_MODE], TITLE,
                        axis=axis)


def run_sweep(out_dir, axis):
    rows_df, summary = sc.run_model_sweep(
        _evaluate, DELETION_PROBS, axis=axis, n_perturb_seeds=N_DELETION_SEEDS,
        clean_mag=0.0, seeds=MODEL_SEEDS, ckpt=CKPT, forward_version=FORWARD_VERSION)
    if summary is None:
        print("[error] no results.")
        return
    sc.save_tables(rows_df, summary, out_dir, STEM)
    make_plots(summary, out_dir, axis)
    sc.print_summary(summary, "HAR deletion sweep", axis)
    print(f"\nFigures (SVG) + CSVs written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/deletion")
    ap.add_argument("--plots-only", action="store_true",
                    help="Regenerate figures from an existing deletion_summary.csv.")
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
