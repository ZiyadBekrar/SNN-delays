"""Temporal-subsampling robustness sweep on HAR.

Keeps 1 of every k timesteps and reconstructs a length-T input, probing whether the
learned delays survive a coarser input clock. k=1 is the identity. ``--recon`` picks
what the network sees between kept samples: ``hold`` repeats the last sample,
``mute`` zeroes the gaps and so also removes most of the drive. Each writes its own
output subtree.

Outputs:  figures/HAR/<sweep>/subsample[_mute]/
Usage:    python experiments/make_figures/HAR/subsample_sweep.py [--recon mute] [--sweep wd]
                 [--plots-only]
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import pandas as pd

from common.subsample import evaluate_subsampled, SUBSAMPLE_XLABEL
import runs as har
import sweep_common as sc

from delrec.utils import calc_metric_HAR

# ======================= TUNE THE SWEEP HERE ============================== #
# Per-value integer subsampling factors k >= 1. 1 = clean input (keep every step).
# On HAR's T=200 these give effective resolutions 200/100/67/50 samples.
SUBSAMPLE_FACTORS = [1, 2, 3, 4]
N_SUBSAMPLE_SEEDS = 1               # deterministic op, one realization per factor
MODEL_SEEDS = har.SEEDS
CKPT = "best.pth"
# Recurrent forward kernel for all model rebuilds. Default 'triton_exact'. All
# kernels are numerically equivalent (see src/delrec/). Alternatives:
# 'v2' (portable pure-torch), 'eventdriven' (fast spike-sparse), None (layer default).
FORWARD_VERSION = "triton_exact"
# ========================================================================= #

# What the gaps between kept samples hold, per --recon: (subsample mode, output
# stem, title). HAR input is a continuous signal, so both modes are "signal" ones.
RECONSTRUCTIONS = {
    "hold": ("signal", "subsample", "HAR: subsampling (hold)"),
    "mute": ("signal_impulse", "subsample_mute", "HAR: subsampling (muted gaps)"),
}


def make_plots(summary, out_dir, axis, stem, title):
    sc.make_sweep_plots(summary, out_dir, stem, SUBSAMPLE_XLABEL, title, axis=axis)


def run_sweep(out_dir, axis, mode, stem, title):
    def evaluate(model, config, device, loader, k, pseed):
        return evaluate_subsampled(loader, model, device, calc_metric_HAR, k, mode, pseed)

    rows_df, summary = sc.run_model_sweep(
        evaluate, SUBSAMPLE_FACTORS, axis=axis, n_perturb_seeds=N_SUBSAMPLE_SEEDS,
        clean_mag=1, seeds=MODEL_SEEDS, ckpt=CKPT, forward_version=FORWARD_VERSION)
    if summary is None:
        print("[error] no results.")
        return
    sc.save_tables(rows_df, summary, out_dir, stem)
    make_plots(summary, out_dir, axis, stem, title)
    sc.print_summary(summary, title, axis)
    print(f"\nFigures (SVG) + CSVs written to {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    har.add_sweep_arg(ap)
    ap.add_argument("--recon", choices=sorted(RECONSTRUCTIONS), default="hold",
                    help="What the gaps between kept samples hold: 'hold' = the last "
                         "sample (zero-order hold, default), 'mute' = zero.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/<sweep>/<subsample|subsample_mute>")
    ap.add_argument("--plots-only", action="store_true",
                    help="Regenerate figures from an existing <stem>_summary.csv.")
    args = ap.parse_args()
    axis = har.get_axis(args)
    mode, stem, title = RECONSTRUCTIONS[args.recon]
    out_dir = args.out_dir or axis.out_dir(stem)
    if args.plots_only:
        summary = sc.load_summary(out_dir, stem, axis)
        make_plots(summary, out_dir, axis, stem, title)
        print(f"Figures (SVG) regenerated in {out_dir}/")
    else:
        run_sweep(out_dir, axis, mode, stem, title)


if __name__ == "__main__":
    main()
